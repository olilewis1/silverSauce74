"""Rich CLI interface for the AI stockbroker agent."""

from __future__ import annotations

import os

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, FloatPrompt, Confirm
from rich import box
from dotenv import load_dotenv

load_dotenv()

from .agent import AgentBrain
from .daemon import TradingDaemon
from .models import AgentConfig, AssetClass, DaemonConfig, RiskLevel, TradeAction, TradeStatus
from .portfolio import PortfolioManager

console = Console()


def _ticker_allowed(asset_class: AssetClass, ticker: str) -> bool:
    normalized = (ticker or "").upper().strip()
    if not normalized:
        return False
    is_crypto = "-USD" in normalized or normalized.endswith("USD")
    if asset_class == AssetClass.STOCKS:
        return not is_crypto
    if asset_class == AssetClass.CRYPTO:
        return is_crypto
    return True


def _portfolio_table(pm: PortfolioManager) -> Table:
    """Build a Rich table showing the portfolio."""
    p = pm.portfolio
    table = Table(title="Portfolio", box=box.ROUNDED, show_lines=True)
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Shares", justify="right")
    table.add_column("Avg Cost", justify="right", style="dim")
    table.add_column("Price", justify="right")
    table.add_column("Value", justify="right", style="green")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")

    for ticker, pos in p.positions.items():
        pnl = pos.unrealised_pnl
        pnl_pct = pos.unrealised_pnl_pct
        pnl_style = "green" if pnl >= 0 else "red"
        table.add_row(
            ticker,
            f"{pos.shares:.2f}",
            f"${pos.avg_cost:.2f}",
            f"${pos.current_price:.2f}",
            f"${pos.market_value:,.2f}",
            f"[{pnl_style}]${pnl:+,.2f}[/{pnl_style}]",
            f"[{pnl_style}]{pnl_pct:+.2%}[/{pnl_style}]",
        )
    return table


def _trade_table(trades) -> Table:
    """Build a Rich table showing trade results."""
    table = Table(title="Trade Results", box=box.ROUNDED, show_lines=True)
    table.add_column("ID", style="dim")
    table.add_column("Ticker", style="cyan bold")
    table.add_column("Action")
    table.add_column("Shares", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Total", justify="right", style="green")
    table.add_column("Status")
    table.add_column("Reasoning", max_width=50)

    for t in trades:
        action_style = {"buy": "green", "sell": "red", "hold": "yellow"}.get(t.action.value, "")
        status_style = {
            "executed": "green",
            "rejected": "red",
            "pending": "yellow",
        }.get(t.status.value, "")
        table.add_row(
            t.id,
            t.ticker,
            f"[{action_style}]{t.action.value.upper()}[/{action_style}]",
            f"{t.shares:.0f}",
            f"${t.price:.2f}" if t.price else "—",
            f"${t.total:,.2f}" if t.total else "—",
            f"[{status_style}]{t.status.value.upper()}[/{status_style}]",
            t.reasoning[:80],
        )
    return table


@click.command()
def main():
    """AI Stockbroker Agent — interactive CLI."""
    console.print(
        Panel.fit(
            "[bold cyan]AI Stockbroker Agent[/bold cyan]\n"
            "Your AI-powered trading assistant.",
            border_style="cyan",
        )
    )

    # --- Trading mode ---
    mode_choice = Prompt.ask(
        "Trading mode",
        choices=["paper", "live"],
        default="paper",
    )
    is_live = mode_choice == "live"

    if is_live:
        console.print(
            Panel(
                "[bold red]⚠  LIVE TRADING MODE ⚠[/bold red]\n\n"
                "This will execute REAL trades with REAL money\n"
                "through your Alpaca brokerage account.\n\n"
                "Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY\n"
                "are set in your .env file.",
                border_style="red",
            )
        )
        if not Confirm.ask("[bold red]Are you sure you want to trade with real money?[/bold red]"):
            console.print("[dim]Switched to paper mode.[/dim]")
            is_live = False

    # --- Setup ---
    risk_choice = Prompt.ask(
        "Risk tolerance",
        choices=["conservative", "moderate", "aggressive"],
        default="moderate",
    )
    risk_level = RiskLevel(risk_choice)

    asset_choice = Prompt.ask(
        "Trade which asset class",
        choices=["stocks", "crypto", "both"],
        default="stocks",
    )
    asset_class = AssetClass(asset_choice)

    max_pct_input = Prompt.ask(
        "Max % of cash to invest (e.g. 60 for 60%, or 'default' for risk-profile default)",
        default="default",
    )
    max_pct = None
    if max_pct_input != "default":
        max_pct = float(max_pct_input) / 100.0

    max_amount_input = Prompt.ask(
        "Hard dollar cap on total investment (or 'none')",
        default="none",
    )
    max_amount = None
    if max_amount_input != "none":
        max_amount = float(max_amount_input)

    config = AgentConfig(
        risk_level=risk_level,
        asset_class=asset_class,
        max_investment_pct=max_pct,
        max_investment_amount=max_amount,
    )

    if is_live:
        pm = PortfolioManager(config, live=True)
        initial_cash = pm.portfolio.initial_cash
        mode_label = "[bold red]LIVE[/bold red]"
    else:
        # Paper mode: sync from Alpaca paper account (ignore user cash input for now)
        # This ensures we use the actual Alpaca paper account state
        pm = PortfolioManager(config, initial_cash=0)  # Will sync from Alpaca
        initial_cash = pm.portfolio.initial_cash
        mode_label = "[bold green]PAPER[/bold green]"
        console.print(f"[dim]Synced from Alpaca paper account: ${initial_cash:,.2f}[/dim]")

    brain = AgentBrain(config, pm)

    profile = config.risk_profile
    console.print(
        Panel(
            f"Mode: {mode_label}\n"
            f"Asset class: [cyan]{config.asset_class.value}[/cyan]\n"
            f"[bold]{profile['description']}[/bold]\n\n"
            f"Max portfolio allocation: [cyan]{config.effective_max_portfolio_pct:.0%}[/cyan]\n"
            f"Max single position: [cyan]{config.effective_max_single_position_pct:.0%}[/cyan]\n"
            f"Available cash: [green]${initial_cash:,.2f}[/green]"
            + (f"\nHard cap: [yellow]${max_amount:,.2f}[/yellow]" if max_amount else ""),
            title="Agent Configuration",
            border_style="green",
        )
    )

    # --- Main loop ---
    while True:
        console.print()
        action = Prompt.ask(
            "[bold cyan]Command[/bold cyan]",
            choices=[
                "analyse",
                "suggest",
                "auto",
                "daemon",
                "buy",
                "sell",
                "portfolio",
                "history",
                "refresh",
                "deposit",
                "quit",
            ],
        )

        if action == "quit":
            console.print("[dim]Goodbye![/dim]")
            break

        elif action == "analyse":
            ticker = Prompt.ask("Ticker to analyse").upper()
            if not _ticker_allowed(config.asset_class, ticker):
                console.print(f"[red]{ticker} is not allowed for asset class mode '{config.asset_class.value}'.[/red]")
                continue
            with console.status(f"Analysing {ticker}..."):
                analysis = brain.analyse_stock(ticker)
            rec_style = {"buy": "green", "sell": "red", "hold": "yellow"}[analysis.recommendation.value]
            console.print(
                Panel(
                    f"[bold]{analysis.ticker}[/bold]\n\n"
                    f"Recommendation: [{rec_style}]{analysis.recommendation.value.upper()}[/{rec_style}]\n"
                    f"Confidence: {analysis.confidence:.0%}\n"
                    f"Risk score: {analysis.risk_score:.0%}\n"
                    f"Target shares: {analysis.target_shares}\n\n"
                    f"[dim]{analysis.reasoning}[/dim]",
                    title="Stock Analysis",
                    border_style=rec_style,
                )
            )

        elif action == "suggest":
            source = Prompt.ask(
                "Suggestion source",
                choices=["ai", "youtube"],
                default="ai",
            )
            if source == "youtube":
                yt_tickers, yt_summaries, _ = _fetch_youtube_strategies(config)
                if yt_summaries:
                    table = Table(title="YouTube Trader Strategies", box=box.ROUNDED)
                    table.add_column("Channel", style="cyan")
                    table.add_column("Strategy", max_width=50)
                    table.add_column("Bullish", style="green")
                    table.add_column("Bearish", style="red")
                    table.add_column("Confidence", style="yellow")
                    for s in yt_summaries:
                        table.add_row(
                            f"@{s.get('channel', '?')}",
                            s.get("strategy_summary", "")[:100],
                            ", ".join(s.get("bullish_tickers", [])) or "—",
                            ", ".join(s.get("bearish_tickers", [])) or "—",
                            f"{s.get('confidence', 0):.0%}",
                        )
                    console.print(table)
                else:
                    console.print("[yellow]No YouTube strategies found.[/yellow]")
            else:
                with console.status("Getting AI suggestions..."):
                    suggestions = brain.suggest_stocks()
                table = Table(title="AI Suggestions", box=box.ROUNDED)
                table.add_column("Ticker", style="cyan bold")
                table.add_column("Reason")
                for s in suggestions:
                    table.add_row(s.get("ticker", "?"), s.get("reason", ""))
                console.print(table)

        elif action == "auto":
            tickers_input = Prompt.ask(
                "Tickers to auto-invest (comma-separated, or 'ai' for AI picks)",
                default="ai",
            )
            tickers = None
            if tickers_input != "ai":
                tickers = [
                    t.strip().upper()
                    for t in tickers_input.split(",")
                    if _ticker_allowed(config.asset_class, t.strip().upper())
                ]

            if Confirm.ask("Proceed with auto-invest?"):
                with console.status("AI agent is working..."):
                    trades = brain.auto_invest(tickers)
                console.print(_trade_table(trades))

        elif action == "daemon":
            _start_daemon(config, pm)

        elif action == "buy":
            ticker = Prompt.ask("Ticker").upper()
            if not _ticker_allowed(config.asset_class, ticker):
                console.print(f"[red]{ticker} is not allowed for asset class mode '{config.asset_class.value}'.[/red]")
                continue
            shares = int(FloatPrompt.ask("Number of shares"))
            with console.status(f"Fetching price for {ticker}..."):
                from .market import get_quote
                quote = get_quote(ticker)
            console.print(f"Current price: [green]${quote.price:.2f}[/green]")
            if Confirm.ask(f"Buy {shares} shares of {ticker} at ${quote.price:.2f}?"):
                from .models import Trade
                trade = Trade(
                    ticker=ticker,
                    action=TradeAction.BUY,
                    shares=shares,
                    price=quote.price,
                    reasoning="Manual buy order.",
                )
                result = pm.execute_trade(trade)
                console.print(_trade_table([result]))

        elif action == "sell":
            ticker = Prompt.ask("Ticker").upper()
            if not _ticker_allowed(config.asset_class, ticker):
                console.print(f"[red]{ticker} is not allowed for asset class mode '{config.asset_class.value}'.[/red]")
                continue
            if ticker not in pm.portfolio.positions:
                console.print(f"[red]No position in {ticker}.[/red]")
                continue
            pos = pm.portfolio.positions[ticker]
            console.print(f"You hold {pos.shares:.2f} shares of {ticker}.")
            shares = int(FloatPrompt.ask("Shares to sell"))
            with console.status(f"Fetching price for {ticker}..."):
                from .market import get_quote
                quote = get_quote(ticker)
            if Confirm.ask(f"Sell {shares} shares of {ticker} at ${quote.price:.2f}?"):
                from .models import Trade
                trade = Trade(
                    ticker=ticker,
                    action=TradeAction.SELL,
                    shares=shares,
                    price=quote.price,
                    reasoning="Manual sell order.",
                )
                result = pm.execute_trade(trade)
                console.print(_trade_table([result]))

        elif action == "portfolio":
            pm.refresh_prices()
            p = pm.portfolio
            console.print(_portfolio_table(pm))
            console.print(
                Panel(
                    f"Cash: [green]${p.cash:,.2f}[/green]\n"
                    f"Invested: [cyan]${p.total_invested:,.2f}[/cyan] "
                    f"({p.invested_pct:.1%} of initial)\n"
                    f"Total value: [bold]${p.total_value:,.2f}[/bold]\n"
                    f"Total P&L: {'[green]' if p.total_pnl >= 0 else '[red]'}"
                    f"${p.total_pnl:+,.2f}{'[/green]' if p.total_pnl >= 0 else '[/red]'}",
                    title="Summary",
                    border_style="blue",
                )
            )

        elif action == "history":
            if not pm.portfolio.trade_history:
                console.print("[dim]No trades yet.[/dim]")
            else:
                console.print(_trade_table(pm.portfolio.trade_history))

        elif action == "refresh":
            with console.status("Refreshing prices..."):
                pm.refresh_prices()
            console.print("[green]Prices updated.[/green]")

        elif action == "deposit":
            amount = FloatPrompt.ask("Amount to deposit ($)")
            pm.deposit(amount)
            console.print(f"[green]Deposited ${amount:,.2f}. New cash: ${pm.portfolio.cash:,.2f}[/green]")


def _get_default_youtube_handles() -> list[str]:
    """Read YOUTUBE_CHANNELS from .env, return as a list of handles (no @ prefix)."""
    raw = os.getenv("YOUTUBE_CHANNELS", "")
    return [h.strip().lstrip("@") for h in raw.split(",") if h.strip()]


def _fetch_youtube_strategies(config: AgentConfig) -> tuple[list[str], list[dict], list[str]]:
    """Prompt for (or use default) YouTube handles and fetch strategies.

    Returns (bullish_tickers, summaries, handles_used).
    """
    from .youtube_strategy import YouTubeStrategyFetcher

    defaults = _get_default_youtube_handles()
    default_display = ", ".join(f"@{h}" for h in defaults) if defaults else "none saved"
    console.print(
        Panel(
            "[bold yellow]YouTube Trader Strategy Mode[/bold yellow]\n\n"
            "The AI will fetch recent video transcripts and extract their top picks.\n\n"
            f"[dim]Default channels (from YOUTUBE_CHANNELS in .env): {default_display}[/dim]\n"
            "[dim]Leave blank to use defaults, or enter new handles.[/dim]",
            border_style="yellow",
        )
    )

    handles_input = Prompt.ask(
        "YouTube handles (comma-separated, or Enter to use defaults)",
        default="",
    )
    if handles_input.strip():
        handles = [h.strip().lstrip("@") for h in handles_input.split(",") if h.strip()]
    else:
        handles = defaults

    if not handles:
        console.print("[red]No handles configured. Set YOUTUBE_CHANNELS in .env or enter handles above.[/red]")
        return [], [], []

    fetcher = YouTubeStrategyFetcher()
    with console.status(f"[yellow]Fetching strategies from {len(handles)} channel(s)...[/yellow]"):
        try:
            tickers, summaries = fetcher.get_tickers_from_channels(handles)
        except RuntimeError as e:
            console.print(f"[red]Error: {e}[/red]")
            return [], [], []

    if not tickers:
        console.print("[yellow]No tickers extracted from YouTube strategies.[/yellow]")

    return tickers, summaries, handles


def _setup_youtube_watchlist(config: AgentConfig) -> tuple[list[str], list[str]]:
    """Prompt for YouTube channel handles and return (watchlist_tickers, channel_handles)."""
    tickers, summaries, used_handles = _fetch_youtube_strategies(config)

    if summaries:
        table = Table(title="YouTube Trader Strategies", box=box.ROUNDED)
        table.add_column("Channel", style="cyan")
        table.add_column("Strategy", style="white", max_width=50)
        table.add_column("Bullish", style="green")
        table.add_column("Bearish", style="red")
        table.add_column("Confidence", style="yellow")
        for s in summaries:
            table.add_row(
                f"@{s.get('channel', '?')}",
                s.get("strategy_summary", "")[:100],
                ", ".join(s.get("bullish_tickers", [])) or "—",
                ", ".join(s.get("bearish_tickers", [])) or "—",
                f"{s.get('confidence', 0):.0%}",
            )
        console.print(table)

    if not tickers:
        console.print("[yellow]No tickers extracted. Falling back to AI picks.[/yellow]")
        return [], []

    console.print(f"\n[green]Watchlist from YouTube strategies:[/green] {', '.join(tickers)}")
    return tickers, used_handles


def _start_daemon(config: AgentConfig, pm: PortfolioManager) -> None:
    """Configure and launch the autonomous trading daemon."""
    console.print(
        Panel(
            "[bold cyan]Autonomous Trading Setup[/bold cyan]\n\n"
            "The daemon will run continuously, analysing configured assets and\n"
            "executing trades on your behalf within your risk limits.\n"
            "The AI will set per-position stop-loss and take-profit levels\n"
            "based on each stock's volatility and market conditions.\n\n"
            "[dim]You can stop it at any time with Ctrl+C.[/dim]",
            border_style="cyan",
        )
    )

    interval = int(FloatPrompt.ask("Check interval in minutes", default=5.0))
    max_day = int(FloatPrompt.ask("Max trades per day", default=100.0))
    max_cycle = int(FloatPrompt.ask("Max trades per cycle", default=5.0))

    watchlist_input = Prompt.ask(
        "Watchlist tickers (comma-separated, 'ai' for AI picks, or 'youtube' to copy a trader's strategy)",
        default="ai",
    )
    watchlist: list[str] = []
    youtube_channels: list[str] = []
    if watchlist_input == "youtube":
        watchlist, youtube_channels = _setup_youtube_watchlist(config)
    elif watchlist_input != "ai":
        watchlist = [
            t.strip().upper()
            for t in watchlist_input.split(",")
            if t.strip() and _ticker_allowed(config.asset_class, t.strip().upper())
        ]

    market_hours = Confirm.ask("Only trade during US market hours?", default=True)

    stop_loss = float(Prompt.ask(
        "Default stop-loss % fallback (AI sets per-position, e.g. 8 for 8%)",
        default="8",
    )) / 100.0

    take_profit = float(Prompt.ask(
        "Default take-profit % fallback (AI sets per-position, e.g. 25 for 25%)",
        default="25",
    )) / 100.0

    cooldown = int(FloatPrompt.ask("Cooldown after trade (minutes)", default=15.0))

    youtube_refresh = 6.0
    if youtube_channels:
        youtube_refresh = float(Prompt.ask(
            "Re-fetch YouTube strategies every N hours",
            default="6",
        ))

    daemon_config = DaemonConfig(
        check_interval_minutes=interval,
        max_trades_per_day=max_day,
        max_trades_per_cycle=max_cycle,
        watchlist=watchlist,
        market_hours_only=market_hours,
        sell_stop_loss_pct=stop_loss,
        sell_take_profit_pct=take_profit,
        cooldown_after_trade_minutes=cooldown,
        youtube_channels=youtube_channels,
        youtube_refresh_hours=youtube_refresh,
    )

    if not Confirm.ask("\n[bold]Start autonomous trading now?[/bold]"):
        console.print("[dim]Cancelled.[/dim]")
        return

    daemon = TradingDaemon(config, daemon_config, pm)
    daemon.start()


if __name__ == "__main__":
    main()
