"""Risk engine — enforces investment constraints and position sizing."""

from __future__ import annotations

import logging
log = logging.getLogger(__name__)

from .models import AgentConfig, Portfolio, StockAnalysis, TradeAction


class RiskEngine:
    """Validates proposed trades against the agent's risk constraints."""

    def __init__(self, config: AgentConfig, portfolio: Portfolio):
        self.config = config
        self.portfolio = portfolio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_buy(self, ticker: str, shares: int, price: float, buying_power: float = None, is_live: bool = False) -> tuple[bool, str, int]:
        """Check if a BUY order is allowed.

        Returns (allowed, reason, adjusted_shares).
        adjusted_shares may be lower than requested if constraints cap it.
        
        Args:
            buying_power: If provided (from Alpaca), use this instead of cash for margin accounts.
            is_live: True if live trading mode (stricter checks).
        """
        cost = shares * price
        
        # Use buying power if provided (handles margin trading), otherwise use cash
        cash_available = buying_power if buying_power is not None else self.portfolio.cash
        
        if is_live:
            # LIVE MODE: Strict checks - never exceed max investable
            max_investable = self._max_investable_cash()
            remaining_investable = max_investable - self.portfolio.total_invested
            if remaining_investable <= 0:
                return False, "Investment cap reached — no more cash can be deployed.", 0
        else:
            # PAPER MODE: Use Alpaca buying power directly (allows margin trading)
            remaining_investable = cash_available

        # Cap by dollar limit (applies to both modes)
        if self.config.max_investment_amount is not None:
            hard_cap_remaining = self.config.max_investment_amount - self.portfolio.total_invested
            remaining_investable = min(remaining_investable, hard_cap_remaining)
            if remaining_investable <= 0:
                return False, "Hard dollar investment cap reached.", 0

        # Cap by single-position limit (use buying power for paper mode)
        if is_live:
            max_position_value = self._max_single_position_value()
        else:
            # Paper mode: base position limit on buying power, not negative cash
            max_position_value = cash_available * self.config.effective_max_single_position_pct
        
        current_position_value = 0.0
        if ticker in self.portfolio.positions:
            current_position_value = self.portfolio.positions[ticker].market_value
        position_room = max_position_value - current_position_value

        allowed_spend = min(remaining_investable, position_room, cash_available)

        if allowed_spend <= 0:
            return False, "Position size limit or insufficient cash.", 0

        if cost > allowed_spend:
            adjusted_shares = int(allowed_spend // price)
            if adjusted_shares <= 0:
                return False, "Adjusted position too small after constraints.", 0
            return (
                True,
                f"Reduced from {shares} to {adjusted_shares} shares to respect limits.",
                adjusted_shares,
            )

        return True, "Trade approved.", shares

    def check_sell(self, ticker: str, shares: int) -> tuple[bool, str, int]:
        """Check if a SELL order is allowed."""
        if ticker not in self.portfolio.positions:
            return False, f"No position in {ticker}.", 0
        held = self.portfolio.positions[ticker].shares
        if shares > held:
            return True, f"Reduced to {int(held)} (all held shares).", int(held)
        return True, "Trade approved.", shares

    def evaluate_analysis(self, analysis: StockAnalysis) -> tuple[bool, str]:
        """Check whether an AI-generated analysis fits risk profile.

        Thresholds per risk level:
        - conservative: reject risk > 0.7, require confidence >= 0.5
        - moderate:     reject risk > 0.85, require confidence >= 0.4
        - aggressive:   reject risk > 0.95, require confidence >= 0.3
        """
        risk_limits = {
            "conservative": {"max_risk": 0.70, "min_confidence": 0.50},
            "moderate":     {"max_risk": 0.85, "min_confidence": 0.40},
            "aggressive":   {"max_risk": 0.95, "min_confidence": 0.30},
        }
        limits = risk_limits.get(self.config.risk_level.value, risk_limits["moderate"])

        if analysis.risk_score > limits["max_risk"]:
            return False, (
                f"Risk score {analysis.risk_score:.0%} exceeds "
                f"{limits['max_risk']:.0%} limit for {self.config.risk_level.value} profile."
            )

        if analysis.confidence < limits["min_confidence"]:
            return False, (
                f"Confidence {analysis.confidence:.0%} below "
                f"{limits['min_confidence']:.0%} minimum for {self.config.risk_level.value} profile."
            )

        return True, "Analysis within risk tolerance."

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _max_investable_cash(self) -> float:
        return self.portfolio.initial_cash * self.config.effective_max_portfolio_pct

    def _max_single_position_value(self) -> float:
        return self.portfolio.initial_cash * self.config.effective_max_single_position_pct
