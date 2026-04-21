"""Data models for the AI stockbroker agent."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class AssetClass(str, Enum):
    STOCKS = "stocks"
    CRYPTO = "crypto"
    BOTH = "both"


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TradeStatus(str, Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Risk profile defaults per level
# ---------------------------------------------------------------------------

RISK_PROFILES = {
    RiskLevel.CONSERVATIVE: {
        "max_portfolio_pct": 0.40,       # invest at most 40% of total funds
        "max_single_position_pct": 0.10, # max 10% in any single stock
        "preferred_sectors": ["utilities", "consumer_staples", "healthcare"],
        "avoid_penny_stocks": True,
        "min_market_cap_b": 10.0,        # only large-cap ($10B+)
        "max_volatility": 0.25,          # annualised vol cap
        "description": "Low risk — focuses on blue-chip, dividend-paying stocks with minimal volatility.",
    },
    RiskLevel.MODERATE: {
        "max_portfolio_pct": 0.65,
        "max_single_position_pct": 0.20,
        "preferred_sectors": ["technology", "healthcare", "financials", "consumer_discretionary"],
        "avoid_penny_stocks": True,
        "min_market_cap_b": 2.0,
        "max_volatility": 0.45,
        "description": "Balanced — mixes growth and value stocks across sectors.",
    },
    RiskLevel.AGGRESSIVE: {
        "max_portfolio_pct": 0.90,
        "max_single_position_pct": 0.35,
        "preferred_sectors": [],  # no restriction
        "avoid_penny_stocks": False,
        "min_market_cap_b": 0.0,
        "max_volatility": 1.0,
        "description": "High risk — chases growth, small-caps, and momentum plays.",
    },
}


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Configuration for the stockbroker agent."""
    risk_level: RiskLevel = RiskLevel.MODERATE
    asset_class: AssetClass = AssetClass.STOCKS
    max_investment_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override: max fraction of total cash the agent can invest (0-1). "
                    "If None, uses the risk profile default.",
    )
    max_single_position_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override: max fraction of total cash in a single position (0-1).",
    )
    max_investment_amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Hard dollar cap on total invested amount.",
    )

    @property
    def effective_max_portfolio_pct(self) -> float:
        if self.max_investment_pct is not None:
            return self.max_investment_pct
        return RISK_PROFILES[self.risk_level]["max_portfolio_pct"]

    @property
    def effective_max_single_position_pct(self) -> float:
        if self.max_single_position_pct is not None:
            return self.max_single_position_pct
        return RISK_PROFILES[self.risk_level]["max_single_position_pct"]

    @property
    def risk_profile(self) -> dict:
        return RISK_PROFILES[self.risk_level]


class Position(BaseModel):
    """A single stock position in the portfolio."""
    ticker: str
    shares: float
    avg_cost: float
    current_price: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost

    @property
    def unrealised_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealised_pnl / self.cost_basis


class Trade(BaseModel):
    """Record of a single trade."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    ticker: str
    action: TradeAction
    shares: float
    price: float
    total: float = 0.0
    status: TradeStatus = TradeStatus.PENDING
    reasoning: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def model_post_init(self, __context) -> None:
        if self.total == 0.0:
            self.total = self.shares * self.price


class Portfolio(BaseModel):
    """The agent's portfolio state."""
    cash: float = 0.0
    initial_cash: float = 0.0
    buying_power: float = 0.0  # Available buying power from broker (handles margin)
    positions: dict[str, Position] = Field(default_factory=dict)
    trade_history: list[Trade] = Field(default_factory=list)

    @property
    def total_invested(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.cash + self.total_invested

    @property
    def total_pnl(self) -> float:
        """Sum of unrealised P&L across all positions."""
        return sum(p.unrealised_pnl for p in self.positions.values())

    @property
    def invested_pct(self) -> float:
        if self.initial_cash == 0:
            return 0.0
        return self.total_invested / self.initial_cash


class StockAnalysis(BaseModel):
    """AI-generated analysis for a stock."""
    ticker: str
    recommendation: TradeAction
    confidence: float = Field(ge=0.0, le=1.0)
    target_shares: int = 0
    reasoning: str = ""
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    suggested_stop_loss: float = Field(default=0.0, ge=0.0, description="AI-suggested stop-loss % for this position")
    suggested_take_profit: float = Field(default=0.0, ge=0.0, description="AI-suggested take-profit % for this position")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DaemonConfig(BaseModel):
    """Configuration for autonomous trading daemon mode."""
    check_interval_minutes: int = Field(
        default=5,
        ge=1,
        description="How often the daemon runs an analysis cycle (minutes).",
    )
    max_trades_per_day: int = Field(
        default=100,
        ge=1,
        description="Maximum number of trades the daemon can execute per calendar day.",
    )
    max_trades_per_cycle: int = Field(
        default=5,
        ge=1,
        description="Maximum trades per single analysis cycle.",
    )
    watchlist: list[str] = Field(
        default_factory=list,
        description="Tickers to monitor. Empty = let AI suggest.",
    )
    market_hours_only: bool = Field(
        default=True,
        description="Only trade during US market hours (9:30-16:00 ET).",
    )
    sell_stop_loss_pct: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Auto-sell if a position drops this % from cost basis.",
    )
    sell_take_profit_pct: float = Field(
        default=0.25,
        ge=0.0,
        le=5.0,
        description="Auto-sell if a position gains this % from cost basis.",
    )
    cooldown_after_trade_minutes: int = Field(
        default=15,
        ge=0,
        description="Wait this long after a trade before trading the same ticker again.",
    )
