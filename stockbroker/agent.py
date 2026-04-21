"""AI Agent brain — uses OpenAI to analyse stocks and make trading decisions."""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

from .market import get_stock_summary, normalize_ticker
from .models import (
    AgentConfig,
    AssetClass,
    Portfolio,
    StockAnalysis,
    Trade,
    TradeAction,
)
from .portfolio import PortfolioManager
from .risk import RiskEngine

load_dotenv()

SYSTEM_PROMPT = """\
You are an AI stockbroker agent. You analyse stocks and make trading decisions
on behalf of your client based on their risk profile and constraints.

You MUST respond with valid JSON only — no markdown, no explanation outside JSON.

Risk profile: {risk_description}
Allowed asset classes: {asset_class_label}
Max portfolio allocation: {max_portfolio_pct:.0%}
Max single-position allocation: {max_single_pct:.0%}
Current cash: ${cash:,.2f}
Current invested: ${invested:,.2f} ({invested_pct:.1%} of funds)
"""

ANALYSE_PROMPT = """\
Analyse the following asset data and decide whether to BUY, SELL, or HOLD.
Consider the client's risk profile, current market conditions, price trends, and
volatility. Be decisive — if the data supports a trade, recommend it.

For crypto assets (BTC-USD, ETH-USD, DOGE-USD, etc.):
- Crypto trades 24/7 and has higher volatility than stocks.
- Consider on-chain trends, market sentiment, Bitcoin dominance, and macro conditions.
- Stop-loss and take-profit should be wider for crypto (e.g. 10-20% SL, 25-50% TP).
- Fractional shares are supported for crypto.

For stocks:
- Consider earnings, sector trends, technical patterns, and news catalysts.
- Stop-loss and take-profit can be tighter (e.g. 5-10% SL, 15-30% TP).

IMPORTANT:
- risk_score should reflect the ACTUAL risk of THIS specific trade, not a blanket score.
  A blue-chip stock in a stable market might be 0.15-0.35. A volatile small-cap might be 0.6-0.85.
  Crypto is typically 0.4-0.7 depending on the coin. Only use 0.9+ for extreme situations.
- confidence should reflect how sure you are about the recommendation.
  If the data clearly supports the trade, use 0.6-0.9.
- target_shares should be a sensible number based on available cash and position limits.
  If the asset is too expensive for the position limit, set target_shares to 0.
  For expensive assets like BTC, suggest fractional amounts (e.g. 1 share = small position).
- suggested_stop_loss is the % drop at which this position should be sold (e.g. 8.0 for 8%).
- suggested_take_profit is the % gain at which profits should be taken (e.g. 15.0 for 15%).
  Base these on the asset's volatility and your analysis.

Stock data:
{stock_data}

Current portfolio positions:
{positions}

Available cash: ${cash:,.2f}
Max single position value: ${max_position_value:,.2f}

Respond with this exact JSON structure:
{{
  "ticker": "<TICKER>",
  "recommendation": "buy" | "sell" | "hold",
  "confidence": <0.0-1.0>,
  "target_shares": <integer>,
  "reasoning": "<2-3 sentence explanation>",
  "risk_score": <0.0-1.0>,
  "suggested_stop_loss": <percent as float, e.g. 8.0>,
  "suggested_take_profit": <percent as float, e.g. 20.0>
}}
"""

STRATEGY_PROMPT = """\
Given the client's portfolio and risk profile, suggest up to 5 tradeable tickers
from the allowed asset class. Consider current market conditions, diversification,
and the client's risk tolerance.

IMPORTANT: Only suggest tickers that can be purchased with the available position limit.
If a single share costs more than ${max_position_value:.2f}, SKIP that ticker.

Return only a JSON array of ticker symbols. Example:
["AAPL", "MSFT", "GOOGL"]

Constraints:
- Max {max_tickers} tickers
- Must be from {asset_class_label}
- Each ticker must cost LESS than ${max_position_value:.2f} per share
- Prefer liquid, actively traded assets
- Avoid suggesting tickers already in portfolio unless there's a strong reason
Allowed asset classes: {asset_class_label}
Available cash: ${cash:,.2f}

IMPORTANT: Only suggest tickers that are supported by the trading platform:

For stocks: Any major US stock (AAPL, TSLA, NVDA, etc.)

For crypto: ONLY these tickers are supported:
- BTC-USD (Bitcoin)
- ETH-USD (Ethereum) 
- BCH-USD (Bitcoin Cash)
- LTC-USD (Litecoin)
- DOGE-USD (Dogecoin)

DO NOT suggest any other crypto tickers (SOL-USD, ADA-USD, etc.) as they will be rejected.

Respond with this exact JSON structure:
{{
  "suggestions": [
    {{
      "ticker": "<TICKER>",
      "reason": "<1 sentence why>"
    }}
  ]
}}
"""


class AgentBrain:
    """The AI reasoning layer that makes investment decisions."""

    def __init__(self, config: AgentConfig, portfolio_mgr: PortfolioManager):
        self.config = config
        self.pm = portfolio_mgr
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Copy .env.example to .env and add your key."
            )
        self.client = OpenAI(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_stock(self, ticker: str) -> StockAnalysis:
        """Fetch market data and ask the AI to analyse a single stock."""
        summary = get_stock_summary(ticker)
        portfolio = self.pm.portfolio

        positions_str = self._format_positions(portfolio)
        max_pos_value = portfolio.initial_cash * self.config.effective_max_single_position_pct
        prompt = ANALYSE_PROMPT.format(
            stock_data=json.dumps(summary, indent=2, default=str),
            positions=positions_str,
            cash=portfolio.cash,
            max_position_value=max_pos_value,
        )

        raw = self._call_llm(prompt, portfolio)
        analysis = self._parse_analysis(raw, ticker)
        return analysis

    def suggest_stocks(self) -> list[dict]:
        """Ask the AI for stock suggestions based on current portfolio."""
        portfolio = self.pm.portfolio
        positions_str = self._format_positions(portfolio)
        profile = self.config.risk_profile

        prompt = STRATEGY_PROMPT.format(
            positions=positions_str,
            risk_description=profile["description"],
            asset_class_label=self._asset_class_label(),
            cash=portfolio.cash,
            max_position_value=portfolio.cash * self.config.effective_max_single_position_pct,
            max_tickers=5,
        )

        raw = self._call_llm(prompt, portfolio)
        try:
            cleaned = self._extract_json(raw)
            data = json.loads(cleaned)
            suggestions: list[dict] = []
            seen: set[str] = set()
            for suggestion in data.get("suggestions", []):
                ticker = normalize_ticker(suggestion.get("ticker", ""))
                if not ticker or not self._ticker_allowed(ticker) or ticker in seen:
                    continue
                seen.add(ticker)
                suggestions.append({
                    "ticker": ticker,
                    "reason": suggestion.get("reason", ""),
                })
            return suggestions
        except (json.JSONDecodeError, ValueError):
            return []

    def auto_invest(self, tickers: Optional[list[str]] = None) -> list[Trade]:
        """Analyse tickers (or get suggestions) and execute approved trades."""
        if not tickers:
            suggestions = self.suggest_stocks()
            tickers = [s["ticker"] for s in suggestions]

        normalized_tickers: list[str] = []
        seen: set[str] = set()
        for ticker in tickers or []:
            normalized = normalize_ticker(ticker)
            if not normalized or not self._ticker_allowed(normalized) or normalized in seen:
                continue
            seen.add(normalized)
            normalized_tickers.append(normalized)

        if not normalized_tickers:
            return [
                Trade(
                    ticker="N/A",
                    action=TradeAction.HOLD,
                    shares=0,
                    price=0,
                    reasoning="No valid tickers were available for auto-invest.",
                )
            ]

        trades: list[Trade] = []
        risk = RiskEngine(self.config, self.pm.portfolio)

        for ticker in normalized_tickers:
            try:
                analysis = self.analyse_stock(ticker)
            except Exception as e:
                trades.append(
                    Trade(
                        ticker=ticker,
                        action=TradeAction.HOLD,
                        shares=0,
                        price=0,
                        reasoning=f"Analysis failed: {e}",
                    )
                )
                continue

            ok, reason = risk.evaluate_analysis(analysis)
            if not ok:
                trades.append(
                    Trade(
                        ticker=ticker,
                        action=TradeAction.HOLD,
                        shares=0,
                        price=0,
                        reasoning=f"Risk check failed: {reason}",
                    )
                )
                continue

            if analysis.recommendation == TradeAction.HOLD:
                trades.append(
                    Trade(
                        ticker=ticker,
                        action=TradeAction.HOLD,
                        shares=0,
                        price=0,
                        reasoning=analysis.reasoning,
                    )
                )
                continue

            from .market import get_quote

            quote = get_quote(ticker)
            
            # Skip if AI suggests 0 shares (asset too expensive)
            if analysis.target_shares <= 0:
                trades.append(
                    Trade(
                        ticker=ticker.upper(),
                        action=TradeAction.HOLD,
                        shares=0,
                        price=quote.price,
                        reasoning=f"Asset too expensive for position limit (${quote.price:.2f} > max position ${self.config.effective_max_single_position_pct * portfolio.cash:.2f})",
                    )
                )
                continue
            
            trade = Trade(
                ticker=ticker.upper(),
                action=analysis.recommendation,
                shares=analysis.target_shares,
                price=quote.price,
                reasoning=analysis.reasoning,
            )
            executed = self.pm.execute_trade(trade)
            trades.append(executed)

        return trades

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str, portfolio: Portfolio) -> str:
        profile = self.config.risk_profile
        asset_labels = {
            "stocks": "stocks only",
            "crypto": "crypto only", 
            "both": "stocks and crypto"
        }
        system = SYSTEM_PROMPT.format(
            risk_description=profile["description"],
            asset_class_label=asset_labels[self.config.asset_class.value],
            max_portfolio_pct=self.config.effective_max_portfolio_pct,
            max_single_pct=self.config.effective_max_single_position_pct,
            cash=portfolio.cash,
            invested=portfolio.total_invested,
            invested_pct=portfolio.invested_pct,
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _parse_analysis(self, raw: str, fallback_ticker: str) -> StockAnalysis:
        try:
            cleaned = self._extract_json(raw)
            data = json.loads(cleaned)
            return StockAnalysis(
                ticker=data.get("ticker", fallback_ticker).upper(),
                recommendation=TradeAction(data["recommendation"].lower()),
                confidence=float(data.get("confidence", 0.5)),
                target_shares=int(data.get("target_shares", 0)),
                reasoning=data.get("reasoning", ""),
                risk_score=float(data.get("risk_score", 0.5)),
                suggested_stop_loss=float(data.get("suggested_stop_loss", 0.0)),
                suggested_take_profit=float(data.get("suggested_take_profit", 0.0)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return StockAnalysis(
                ticker=fallback_ticker.upper(),
                recommendation=TradeAction.HOLD,
                confidence=0.0,
                target_shares=0,
                reasoning=f"Failed to parse AI response: {e} — {raw[:200]}",
                risk_score=1.0,
            )

    @staticmethod
    def _extract_json(text: str) -> str:
        """Strip markdown code fences from AI responses to get raw JSON."""
        text = text.strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def _asset_class_label(self) -> str:
        if self.config.asset_class == AssetClass.STOCKS:
            return "stocks only"
        if self.config.asset_class == AssetClass.CRYPTO:
            return "crypto only"
        return "stocks and crypto"

    def _ticker_allowed(self, ticker: str) -> bool:
        normalized = normalize_ticker(ticker)
        if not normalized:
            return False
        is_crypto = "-USD" in normalized or normalized.endswith("USD")
        if self.config.asset_class == AssetClass.STOCKS:
            return not is_crypto
        if self.config.asset_class == AssetClass.CRYPTO:
            return is_crypto
        return True

    @staticmethod
    def _format_positions(portfolio: Portfolio) -> str:
        if not portfolio.positions:
            return "No current positions."
        lines = []
        for ticker, pos in portfolio.positions.items():
            lines.append(
                f"  {ticker}: {pos.shares} shares @ ${pos.avg_cost:.2f} "
                f"(current: ${pos.current_price:.2f}, P&L: {pos.unrealised_pnl_pct:.1%})"
            )
        return "\n".join(lines)
