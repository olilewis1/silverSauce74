"""Autonomous trading daemon — runs continuously, analysing and trading on a schedule."""

from __future__ import annotations

import logging
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich import box

from .agent import AgentBrain
from .market import get_quote, normalize_ticker
from .models import (
    AgentConfig,
    AssetClass,
    DaemonConfig,
    Trade,
    TradeAction,
    TradeStatus,
)
from .portfolio import PortfolioManager
from .risk import RiskEngine

ET = ZoneInfo("America/New_York")
LOG_FILE = Path("daemon.log")

console = Console()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("stockbroker.daemon")
    logger.setLevel(logging.INFO)

    # Rich console handler
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
    )
    console_handler.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# ---------------------------------------------------------------------------
# Daemon class
# ---------------------------------------------------------------------------

class TradingDaemon:
    """Autonomous trading loop that runs on a schedule."""

    def __init__(
        self,
        agent_config: AgentConfig,
        daemon_config: DaemonConfig,
        portfolio_mgr: PortfolioManager,
    ):
        self.agent_config = agent_config
        self.daemon_config = daemon_config
        self.pm = portfolio_mgr
        self.brain = AgentBrain(agent_config, portfolio_mgr)
        self.log = _setup_logging()

        self._running = False
        self._trades_today: list[Trade] = []
        self._today: Optional[str] = None
        self._cooldowns: dict[str, datetime] = {}  # ticker -> earliest next trade time
        self._cycle_count = 0
        self._exit_levels: dict[str, dict] = {}  # ticker -> {stop_loss, take_profit}
        self._youtube_bearish: set[str] = set()  # tickers flagged bearish by YouTubers
        self._last_youtube_refresh: Optional[datetime] = None

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the autonomous trading loop."""
        self._running = True
        self._print_banner()
        self.log.info("Daemon started. Interval: %d min", self.daemon_config.check_interval_minutes)

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log.error("Cycle error: %s", e, exc_info=True)

            if not self._running:
                break

            wait_sec = self.daemon_config.check_interval_minutes * 60
            self.log.info(
                "Sleeping %d minutes until next cycle...",
                self.daemon_config.check_interval_minutes,
            )
            self._interruptible_sleep(wait_sec)

        self.log.info("Daemon stopped.")
        self._print_summary()

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Run a single analysis + trade cycle."""
        self._cycle_count += 1
        now = datetime.now(ET)
        today_str = now.strftime("%Y-%m-%d")

        # Reset daily trade counter
        if self._today != today_str:
            self._today = today_str
            self._trades_today = []

        self.log.info("=== Cycle #%d at %s ET ===", self._cycle_count, now.strftime("%H:%M:%S"))

        # YouTube strategy refresh (if channels are configured)
        self._maybe_refresh_youtube()

        # Market hours check
        if self._should_enforce_market_hours() and not self._is_market_open(now):
            self.log.info("[yellow]Market closed.[/yellow] Skipping cycle.")
            return

        # Daily trade cap
        if len(self._trades_today) >= self.daemon_config.max_trades_per_day:
            self.log.info(
                "[yellow]Daily trade limit reached (%d/%d).[/yellow] Skipping cycle.",
                len(self._trades_today),
                self.daemon_config.max_trades_per_day,
            )
            return

        # 1. Refresh portfolio prices
        self.log.info("Refreshing portfolio prices...")
        self.pm.refresh_prices()

        # 2. Check stop-loss / take-profit on existing positions
        self._check_exit_signals()

        # 3. Get tickers to analyse
        tickers = self._get_tickers()
        if not tickers:
            self.log.info("No tickers to analyse this cycle.")
            return

        # 4. Analyse and trade
        trades_this_cycle = 0
        remaining_daily = self.daemon_config.max_trades_per_day - len(self._trades_today)
        max_this_cycle = min(self.daemon_config.max_trades_per_cycle, remaining_daily)

        for ticker in tickers:
            if trades_this_cycle >= max_this_cycle:
                self.log.info("Cycle trade limit reached (%d).", max_this_cycle)
                break

            if self._is_on_cooldown(ticker):
                self.log.info("[dim]%s on cooldown, skipping.[/dim]", ticker)
                continue

            try:
                trade, analysis = self._analyse_and_trade(ticker)
                if trade and trade.status == TradeStatus.EXECUTED:
                    trades_this_cycle += 1
                    self._trades_today.append(trade)
                    self._set_cooldown(ticker)
                    # Store AI-suggested exit levels for this position
                    if analysis and trade.action == TradeAction.BUY:
                        sl = analysis.suggested_stop_loss
                        tp = analysis.suggested_take_profit
                        if sl > 0 or tp > 0:
                            self._exit_levels[ticker] = {
                                "stop_loss": sl / 100.0 if sl > 0 else self.daemon_config.sell_stop_loss_pct,
                                "take_profit": tp / 100.0 if tp > 0 else self.daemon_config.sell_take_profit_pct,
                            }
                            self.log.info(
                                "AI exit levels for %s: SL=%.1f%% TP=%.1f%%",
                                ticker,
                                self._exit_levels[ticker]["stop_loss"] * 100,
                                self._exit_levels[ticker]["take_profit"] * 100,
                            )
            except Exception as e:
                self.log.error("Error processing %s: %s", ticker, e)

        self.log.info(
            "Cycle #%d complete. Trades this cycle: %d | Today: %d/%d",
            self._cycle_count,
            trades_this_cycle,
            len(self._trades_today),
            self.daemon_config.max_trades_per_day,
        )
        self._log_portfolio_snapshot()

    # ------------------------------------------------------------------
    # Stop-loss / take-profit
    # ------------------------------------------------------------------

    def _check_exit_signals(self) -> None:
        """Auto-sell positions that hit stop-loss or take-profit.

        Uses AI-suggested levels per position when available,
        falls back to global daemon defaults.
        """
        positions = list(self.pm.portfolio.positions.items())
        for ticker, pos in positions:
            if pos.cost_basis == 0:
                continue

            pnl_pct = pos.unrealised_pnl_pct

            # Get per-position AI levels or fall back to global defaults
            levels = self._exit_levels.get(ticker, {})
            stop_loss = levels.get("stop_loss", self.daemon_config.sell_stop_loss_pct)
            take_profit = levels.get("take_profit", self.daemon_config.sell_take_profit_pct)

            # Stop-loss
            if pnl_pct <= -stop_loss:
                self.log.warning(
                    "[red]STOP-LOSS[/red] %s at %.2f%% loss (limit: %.1f%%). Selling %d shares.",
                    ticker, pnl_pct * 100, stop_loss * 100, int(pos.shares),
                )
                self._execute_sell(ticker, int(pos.shares), f"Stop-loss triggered at {pnl_pct:.1%} (limit: {stop_loss:.0%})")

            # Take-profit
            elif pnl_pct >= take_profit:
                self.log.info(
                    "[green]TAKE-PROFIT[/green] %s at +%.2f%% (limit: %.1f%%). Selling %d shares.",
                    ticker, pnl_pct * 100, take_profit * 100, int(pos.shares),
                )
                self._execute_sell(ticker, int(pos.shares), f"Take-profit triggered at {pnl_pct:+.1%} (limit: {take_profit:.0%})")

    def _execute_sell(self, ticker: str, shares: int, reason: str) -> Optional[Trade]:
        """Execute a sell trade for exit signals."""
        try:
            # Normalize ticker for market data (DOGEUSD -> DOGE-USD)
            normalized_ticker = normalize_ticker(ticker)
            quote = get_quote(normalized_ticker)
            trade = Trade(
                ticker=ticker,
                action=TradeAction.SELL,
                shares=shares,
                price=quote.price,
                reasoning=reason,
            )
            result = self.pm.execute_trade(trade)
            if result.status == TradeStatus.EXECUTED:
                self._trades_today.append(result)
                self._set_cooldown(ticker)
                self.log.info("SOLD %d shares of %s @ $%.2f — %s", shares, ticker, quote.price, reason)
            return result
        except Exception as e:
            self.log.error("Failed to sell %s: %s", ticker, e)
            return None

    # ------------------------------------------------------------------
    # AI analysis → trade
    # ------------------------------------------------------------------

    def _analyse_and_trade(self, ticker: str) -> tuple[Optional[Trade], Optional[object]]:
        """Analyse a single ticker and execute if the AI recommends it.

        Returns (trade, analysis) so the caller can store AI-suggested exit levels.
        """
        self.log.info("Analysing [cyan]%s[/cyan]...", ticker)

        analysis = self.brain.analyse_stock(ticker)
        self.log.info(
            "%s → %s (confidence: %.0f%%, risk: %.0f%%)",
            ticker,
            analysis.recommendation.value.upper(),
            analysis.confidence * 100,
            analysis.risk_score * 100,
        )

        # Risk gate
        risk = RiskEngine(self.agent_config, self.pm.portfolio)
        ok, reason = risk.evaluate_analysis(analysis)
        if not ok:
            self.log.info("[yellow]Risk rejected %s: %s[/yellow]", ticker, reason)
            return None, analysis

        if analysis.recommendation == TradeAction.HOLD:
            self.log.info("%s: HOLD — no action.", ticker)
            return None, analysis

        if analysis.recommendation == TradeAction.SELL:
            if ticker not in self.pm.portfolio.positions:
                self.log.info("%s: SELL recommended but no position held.", ticker)
                return None, analysis
            pos = self.pm.portfolio.positions[ticker]
            shares = min(analysis.target_shares, int(pos.shares))
            if shares <= 0:
                shares = int(pos.shares)
            result = self._execute_sell(ticker, shares, analysis.reasoning)
            return result, analysis

        # BUY
        quote = get_quote(ticker)
        trade = Trade(
            ticker=ticker.upper(),
            action=TradeAction.BUY,
            shares=analysis.target_shares,
            price=quote.price,
            reasoning=analysis.reasoning,
        )
        result = self.pm.execute_trade(trade)
        if result.status == TradeStatus.EXECUTED:
            self.log.info(
                "[green]BOUGHT[/green] %d shares of %s @ $%.2f — %s",
                int(result.shares), ticker, result.price, result.reasoning[:80],
            )
        elif result.status == TradeStatus.REJECTED:
            self.log.warning(
                "[red]REJECTED[/red] %s buy: %s", ticker, result.reasoning,
            )
        return result, analysis

    # ------------------------------------------------------------------
    # Ticker selection
    # ------------------------------------------------------------------

    def _ticker_allowed(self, ticker: str) -> bool:
        """Check if ticker matches the configured asset class filter and is supported by Alpaca."""
        from .broker import AlpacaBroker
        is_crypto = AlpacaBroker._is_crypto(ticker)
        
        # Check asset class filter
        if self.agent_config.asset_class == AssetClass.STOCKS:
            if is_crypto:
                return False
        elif self.agent_config.asset_class == AssetClass.CRYPTO:
            if not is_crypto:
                return False
        
        # For crypto, check if Alpaca supports it
        if is_crypto:
            broker = AlpacaBroker()
            if not broker.is_supported_crypto(ticker):
                self.log.warning(
                    "[yellow]%s not supported by Alpaca (only BTC, ETH, BCH, LTC, DOGE, etc)[/yellow]",
                    ticker
                )
                return False
        
        return True

    def _maybe_refresh_youtube(self) -> None:
        """Re-fetch YouTube strategies periodically and update the watchlist."""
        channels = self.daemon_config.youtube_channels
        if not channels:
            return

        refresh_delta = timedelta(hours=self.daemon_config.youtube_refresh_hours)
        if self._last_youtube_refresh and datetime.utcnow() - self._last_youtube_refresh < refresh_delta:
            return

        self.log.info(
            "[yellow]Refreshing YouTube strategies from %d channel(s)...[/yellow]",
            len(channels),
        )
        try:
            from .youtube_strategy import YouTubeStrategyFetcher
            fetcher = YouTubeStrategyFetcher()
            tickers, summaries = fetcher.get_tickers_from_channels(channels)

            # Update bearish blacklist
            new_bearish: set[str] = set()
            for s in summaries:
                new_bearish.update(t.upper() for t in s.get("bearish_tickers", []))
            if new_bearish != self._youtube_bearish:
                self.log.info(
                    "[red]YouTube bearish blacklist updated:[/red] %s",
                    ", ".join(sorted(new_bearish)) or "(none)",
                )
            self._youtube_bearish = new_bearish

            # Update watchlist (filtered)
            new_watchlist = [
                t.upper() for t in tickers
                if self._ticker_allowed(t) and t.upper() not in self._youtube_bearish
            ]
            if new_watchlist:
                self.daemon_config.watchlist = new_watchlist
                self.log.info(
                    "[green]YouTube watchlist updated:[/green] %s",
                    ", ".join(new_watchlist),
                )
            else:
                self.log.warning("YouTube refresh returned no valid tickers.")

            self._last_youtube_refresh = datetime.utcnow()
        except Exception as e:
            self.log.error("YouTube strategy refresh failed: %s", e)

    def _get_tickers(self) -> list[str]:
        """Get the list of tickers to analyse this cycle."""
        if self.daemon_config.watchlist:
            return [
                t.upper() for t in self.daemon_config.watchlist
                if self._ticker_allowed(t) and t.upper() not in self._youtube_bearish
            ]

        # Let the AI suggest
        self.log.info("No watchlist set — asking AI for suggestions...")
        try:
            suggestions = self.brain.suggest_stocks()
            tickers = [
                s["ticker"].upper()
                for s in suggestions
                if s.get("ticker") and self._ticker_allowed(s["ticker"])
            ]
            self.log.info("AI suggested: %s", ", ".join(tickers))
            return tickers
        except Exception as e:
            self.log.error("Failed to get AI suggestions: %s", e)
            return []

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    @staticmethod
    def _is_market_open(now: Optional[datetime] = None) -> bool:
        """Check if US stock market is currently open (rough check, ignores holidays)."""
        if now is None:
            now = datetime.now(ET)
        # Weekday check (Mon=0 .. Fri=4)
        if now.weekday() > 4:
            return False
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close

    def _should_enforce_market_hours(self) -> bool:
        """Check if market hours should be enforced based on asset class."""
        # Crypto trades 24/7, so don't enforce market hours for crypto-only mode
        if self.agent_config.asset_class == AssetClass.CRYPTO:
            return False
        # For stocks or mixed mode, respect the daemon config setting
        return self.daemon_config.market_hours_only

    # ------------------------------------------------------------------
    # Cooldowns
    # ------------------------------------------------------------------

    def _is_on_cooldown(self, ticker: str) -> bool:
        if ticker not in self._cooldowns:
            return False
        return datetime.utcnow() < self._cooldowns[ticker]

    def _set_cooldown(self, ticker: str) -> None:
        self._cooldowns[ticker] = datetime.utcnow() + timedelta(
            minutes=self.daemon_config.cooldown_after_trade_minutes
        )

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        dc = self.daemon_config
        profile = self.agent_config.risk_profile
        watchlist_str = ", ".join(dc.watchlist) if dc.watchlist else "AI-selected"

        console.print(
            Panel(
                f"[bold cyan]AUTONOMOUS TRADING MODE[/bold cyan]\n\n"
                f"Risk profile: [bold]{profile['description']}[/bold]\n"
                f"Asset class: [cyan]{self.agent_config.asset_class.value}[/cyan]\n"
                f"Check interval: [cyan]{dc.check_interval_minutes} min[/cyan]\n"
                f"Max trades/day: [cyan]{dc.max_trades_per_day}[/cyan]\n"
                f"Max trades/cycle: [cyan]{dc.max_trades_per_cycle}[/cyan]\n"
                f"Watchlist: [cyan]{watchlist_str}[/cyan]\n"
                f"Market hours only: [cyan]{self._should_enforce_market_hours()}[/cyan]\n"
                f"Stop-loss: [red]{dc.sell_stop_loss_pct:.0%}[/red]\n"
                f"Take-profit: [green]{dc.sell_take_profit_pct:.0%}[/green]\n"
                f"Cooldown after trade: [cyan]{dc.cooldown_after_trade_minutes} min[/cyan]\n\n"
                f"Cash: [green]${self.pm.portfolio.cash:,.2f}[/green]\n\n"
                f"[dim]Press Ctrl+C to stop.[/dim]",
                border_style="bold cyan",
                title="Daemon Started",
            )
        )

    def _log_portfolio_snapshot(self) -> None:
        p = self.pm.portfolio
        self.log.info(
            "Portfolio: cash=$%.2f | invested=$%.2f (%.1f%%) | total=$%.2f | P&L=$%+.2f",
            p.cash,
            p.total_invested,
            p.invested_pct * 100,
            p.total_value,
            p.total_pnl,
        )

    def _print_summary(self) -> None:
        p = self.pm.portfolio
        console.print()
        console.print(
            Panel(
                f"Cycles run: [cyan]{self._cycle_count}[/cyan]\n"
                f"Trades today: [cyan]{len(self._trades_today)}[/cyan]\n\n"
                f"Cash: [green]${p.cash:,.2f}[/green]\n"
                f"Invested: [cyan]${p.total_invested:,.2f}[/cyan]\n"
                f"Total value: [bold]${p.total_value:,.2f}[/bold]\n"
                f"Total P&L: {'[green]' if p.total_pnl >= 0 else '[red]'}"
                f"${p.total_pnl:+,.2f}{'[/green]' if p.total_pnl >= 0 else '[/red]'}",
                title="Daemon Session Summary",
                border_style="blue",
            )
        )

        if self._trades_today:
            table = Table(title="Today's Trades", box=box.ROUNDED, show_lines=True)
            table.add_column("Time", style="dim")
            table.add_column("Ticker", style="cyan bold")
            table.add_column("Action")
            table.add_column("Shares", justify="right")
            table.add_column("Price", justify="right")
            table.add_column("Total", justify="right")
            table.add_column("Reasoning", max_width=50)
            for t in self._trades_today:
                action_style = {"buy": "green", "sell": "red", "hold": "yellow"}.get(t.action.value, "")
                table.add_row(
                    t.timestamp.strftime("%H:%M"),
                    t.ticker,
                    f"[{action_style}]{t.action.value.upper()}[/{action_style}]",
                    f"{t.shares:.0f}",
                    f"${t.price:.2f}",
                    f"${t.total:,.2f}",
                    t.reasoning[:60],
                )
            console.print(table)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep that can be interrupted by Ctrl+C."""
        end = time.time() + seconds
        while time.time() < end and self._running:
            time.sleep(min(1.0, end - time.time()))

    def _handle_shutdown(self, signum, frame) -> None:
        self.log.info("Shutdown signal received. Finishing current cycle...")
        self._running = False
