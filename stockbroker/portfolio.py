"""Portfolio manager — tracks positions, executes trades via paper or Alpaca live."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .market import get_quote
from .models import AgentConfig, Portfolio, Position, Trade, TradeAction, TradeStatus
from .risk import RiskEngine

STATE_FILE = Path("portfolio_state.json")
log = logging.getLogger("stockbroker.portfolio")


class PortfolioManager:
    """Manages the portfolio and executes trades.

    Supports two modes:
      - paper (default): simulated execution, virtual cash
      - live: real execution via Alpaca, cash synced from broker
    """

    def __init__(
        self,
        config: AgentConfig,
        initial_cash: float = 0.0,
        live: bool = False,
    ):
        self.config = config
        self.live = live
        self._broker = None

        self.portfolio = Portfolio(cash=initial_cash, initial_cash=initial_cash)

        # Always use Alpaca broker (paper or live) so trades appear in dashboard
        from .broker import AlpacaBroker
        self._broker = AlpacaBroker()
        self._sync_from_broker()

    # ------------------------------------------------------------------
    # Cash management
    # ------------------------------------------------------------------

    def deposit(self, amount: float) -> None:
        """Add cash to the portfolio (paper mode only)."""
        if self.live:
            log.warning("Deposit not supported in live mode — fund your Alpaca account directly.")
            return
        self.portfolio.cash += amount
        self.portfolio.initial_cash += amount
        self._save_state()

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_trade(self, trade: Trade) -> Trade:
        """Execute a trade after risk checks. Routes through Alpaca (paper or live)."""
        risk = RiskEngine(self.config, self.portfolio)

        if trade.action == TradeAction.BUY:
            # Get buying power from Alpaca for margin trading support
            buying_power = None
            is_live = False
            if self._broker:
                try:
                    buying_power = self._broker.get_account()['buying_power']
                    is_live = self._broker.is_live
                except:
                    pass
            
            allowed, reason, adj_shares = risk.check_buy(
                trade.ticker, trade.shares, trade.price,
                buying_power=buying_power, is_live=is_live
            )
            if not allowed:
                trade.status = TradeStatus.REJECTED
                trade.reasoning += f" | REJECTED: {reason}"
                self.portfolio.trade_history.append(trade)
                self._save_state()
                return trade

            trade.shares = adj_shares
            trade.total = adj_shares * trade.price

            # Always route through Alpaca (paper or live based on ALPACA_MODE)
            result = self._broker.buy(trade.ticker, adj_shares)
            if result["status"] == "filled":
                # Paper mode may return avg_price=0 for pending orders
                if result["avg_price"] and result["avg_price"] > 0:
                    trade.price = result["avg_price"]
                    trade.total = result["shares"] * result["avg_price"]
                trade.shares = result["shares"]
                trade.status = TradeStatus.EXECUTED
                self._sync_from_broker()
            else:
                trade.status = TradeStatus.REJECTED
                trade.reasoning += f" | BROKER: {result.get('error', result['status'])}"

        elif trade.action == TradeAction.SELL:
            allowed, reason, adj_shares = risk.check_sell(trade.ticker, trade.shares)
            if not allowed:
                trade.status = TradeStatus.REJECTED
                trade.reasoning += f" | REJECTED: {reason}"
                self.portfolio.trade_history.append(trade)
                self._save_state()
                return trade

            trade.shares = adj_shares
            trade.total = adj_shares * trade.price

            # Always route through Alpaca (paper or live based on ALPACA_MODE)
            result = self._broker.sell(trade.ticker, adj_shares)
            if result["status"] == "filled":
                trade.price = result["avg_price"]
                trade.shares = result["shares"]
                trade.total = result["total"]
                trade.status = TradeStatus.EXECUTED
                self._sync_from_broker()
            else:
                trade.status = TradeStatus.REJECTED
                trade.reasoning += f" | BROKER: {result.get('error', result['status'])}"

        else:
            trade.status = TradeStatus.REJECTED
            trade.reasoning += " | HOLD — no trade executed."

        self.portfolio.trade_history.append(trade)
        self._save_state()
        return trade

    # ------------------------------------------------------------------
    # Portfolio refresh
    # ------------------------------------------------------------------

    def refresh_prices(self) -> None:
        """Update all position prices from Alpaca."""
        self._sync_from_broker()

    # ------------------------------------------------------------------
    # Alpaca sync
    # ------------------------------------------------------------------

    def _sync_from_broker(self) -> None:
        """Pull real account state from Alpaca into the local portfolio model."""
        if not self._broker:
            return

        account = self._broker.get_account()
        positions = self._broker.get_positions()

        cash = account["cash"]
        buying_power = account["buying_power"]
        portfolio_value = account["portfolio_value"]
        
        # For margin accounts with negative cash, use portfolio_value as the base
        if self.portfolio.initial_cash == 0:
            if cash < 0:
                # Margin account: use portfolio value as initial capital
                self.portfolio.initial_cash = portfolio_value
            else:
                self.portfolio.initial_cash = cash

        self.portfolio.cash = cash
        # Store buying power for risk calculations
        self.portfolio.buying_power = buying_power

        # Rebuild positions from broker
        new_positions: dict[str, Position] = {}
        for p in positions:
            new_positions[p["ticker"]] = Position(
                ticker=p["ticker"],
                shares=p["shares"],
                avg_cost=p["avg_cost"],
                current_price=p["current_price"],
                last_updated=datetime.utcnow(),
            )
        self.portfolio.positions = new_positions
        self._save_state()
        log.info(
            "Synced from Alpaca: cash=$%.2f, %d positions, equity=$%.2f",
            cash, len(positions), account["portfolio_value"],
        )

    # ------------------------------------------------------------------
    # Paper-mode internals
    # ------------------------------------------------------------------

    def _apply_buy(self, trade: Trade) -> None:
        ticker = trade.ticker
        if ticker in self.portfolio.positions:
            pos = self.portfolio.positions[ticker]
            total_cost = pos.avg_cost * pos.shares + trade.total
            pos.shares += trade.shares
            pos.avg_cost = total_cost / pos.shares if pos.shares else 0
            pos.current_price = trade.price
        else:
            self.portfolio.positions[ticker] = Position(
                ticker=ticker,
                shares=trade.shares,
                avg_cost=trade.price,
                current_price=trade.price,
            )
        self.portfolio.cash -= trade.total

    def _apply_sell(self, trade: Trade) -> None:
        ticker = trade.ticker
        pos = self.portfolio.positions[ticker]
        pos.shares -= trade.shares
        pos.current_price = trade.price
        self.portfolio.cash += trade.total
        if pos.shares <= 0:
            del self.portfolio.positions[ticker]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        STATE_FILE.write_text(self.portfolio.model_dump_json(indent=2))

    def _load_state(self) -> None:
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            self.portfolio = Portfolio.model_validate(data)
