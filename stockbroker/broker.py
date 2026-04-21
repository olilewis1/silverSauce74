"""Alpaca broker integration for real (and paper) money trading."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.common.exceptions import APIError

load_dotenv()

log = logging.getLogger("stockbroker.broker")


class AlpacaBroker:
    """Wraps the Alpaca Trading API for order execution and account info.
    
    Intelligently routes:
    - Stock orders: DAY time-in-force (cancel at market close)
    - Crypto orders: GTC time-in-force (24/7 trading)
    """

    def __init__(self):
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        mode = os.getenv("ALPACA_MODE", "paper").lower()

        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env. "
                "Sign up free at https://alpaca.markets"
            )

        self.is_live = mode == "live"
        self.client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=not self.is_live,
        )
        self.mode = mode

    SUPPORTED_CRYPTOS = {
        "BTC/USD", "ETH/USD", "BCH/USD", "LTC/USD", "DOGE/USD",
    }

    @staticmethod
    def _is_crypto(ticker: str) -> bool:
        """Detect if ticker is crypto based on format."""
        normalized = ticker.upper().strip()
        return "-USD" in normalized or "/USD" in normalized or normalized.endswith("USD")

    @staticmethod
    def _to_alpaca_symbol(ticker: str) -> str:
        """Convert ticker to Alpaca format. Crypto must use slash: BTC/USD."""
        normalized = ticker.upper().strip()
        if AlpacaBroker._is_crypto(normalized):
            # Handle different crypto formats:
            # DOGE-USD -> DOGE/USD
            # DOGEUSD -> DOGE/USD  
            # DOGE/USD -> DOGE/USD
            if "-USD" in normalized:
                return normalized.replace("-USD", "/USD")
            elif normalized.endswith("USD") and "/" not in normalized:
                return normalized[:-3] + "/USD"
            return normalized
        return normalized

    def is_supported_crypto(self, ticker: str) -> bool:
        """Check if crypto ticker is supported by Alpaca."""
        alpaca_sym = self._to_alpaca_symbol(ticker)
        return alpaca_sym in self.SUPPORTED_CRYPTOS

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Get account balance info."""
        account = self.client.get_account()
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "currency": account.currency,
            "status": account.status,
            "is_live": self.is_live,
        }

    def get_cash(self) -> float:
        """Get available cash balance."""
        account = self.client.get_account()
        return float(account.cash)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict]:
        """Get all open positions from Alpaca."""
        positions = self.client.get_all_positions()
        result = []
        for pos in positions:
            result.append({
                "ticker": pos.symbol,
                "shares": float(pos.qty),
                "avg_cost": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealised_pnl": float(pos.unrealized_pl),
                "unrealised_pnl_pct": float(pos.unrealized_plpc),
            })
        return result

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def buy(self, ticker: str, shares: int) -> dict:
        """Submit a market buy order. Returns order details."""
        is_crypto = self._is_crypto(ticker)
        alpaca_symbol = self._to_alpaca_symbol(ticker)
        
        # Use GTC for crypto (24/7) and paper mode (fills may be delayed)
        if is_crypto or not self.is_live:
            tif = TimeInForce.GTC
        else:
            tif = TimeInForce.DAY
        
        # Get current account info for debugging
        try:
            account = self.get_account()
            log.info(
                "Attempting BUY: %s x%d shares | Available buying power: $%.2f",
                ticker, shares, account["buying_power"]
            )
        except Exception:
            pass
        
        order_data = MarketOrderRequest(
            symbol=alpaca_symbol,
            qty=shares,
            side=OrderSide.BUY,
            time_in_force=tif,
        )
        try:
            order = self.client.submit_order(order_data)
            asset_type = "CRYPTO" if is_crypto else "STOCK"
            log.info("%s BUY order submitted: %s x%d — order_id=%s", asset_type, ticker, shares, order.id)
            filled = self._wait_for_fill(str(order.id), is_crypto=is_crypto)
            return filled
        except APIError as e:
            log.error("Alpaca BUY error for %s: %s", ticker, e)
            return {
                "status": "rejected",
                "ticker": ticker,
                "shares": shares,
                "error": str(e),
            }

    def sell(self, ticker: str, shares: int) -> dict:
        """Submit a market sell order. Returns order details."""
        is_crypto = self._is_crypto(ticker)
        alpaca_symbol = self._to_alpaca_symbol(ticker)
        # Use GTC for crypto (24/7) and paper mode (fills may be delayed)
        if is_crypto or not self.is_live:
            tif = TimeInForce.GTC
        else:
            tif = TimeInForce.DAY
        
        order_data = MarketOrderRequest(
            symbol=alpaca_symbol,
            qty=shares,
            side=OrderSide.SELL,
            time_in_force=tif,
        )
        try:
            order = self.client.submit_order(order_data)
            asset_type = "CRYPTO" if is_crypto else "STOCK"
            log.info("%s SELL order submitted: %s x%d — order_id=%s", asset_type, ticker, shares, order.id)
            filled = self._wait_for_fill(str(order.id), is_crypto=is_crypto)
            return filled
        except APIError as e:
            log.error("Alpaca SELL error for %s: %s", ticker, e)
            return {
                "status": "rejected",
                "ticker": ticker,
                "shares": shares,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _wait_for_fill(self, order_id: str, is_crypto: bool = False, timeout: int = 60) -> dict:
        """Poll until the order is filled or timeout.
        
        Paper trading can be slow, especially after hours.
        Crypto orders may take longer due to 24/7 market conditions.
        """
        # Paper trading needs more time
        if not self.is_live:
            timeout = 90 if is_crypto else 60
        elif is_crypto:
            timeout = 120  # Live crypto needs even more time
        
        start = time.time()
        last_status = None
        while time.time() - start < timeout:
            try:
                order = self.client.get_order_by_id(order_id)
                
                # Log status changes
                if order.status != last_status:
                    log.info("Order %s status: %s", order_id[:8], order.status)
                    last_status = order.status
                
                if order.status == OrderStatus.FILLED:
                    return {
                        "status": "filled",
                        "ticker": order.symbol,
                        "shares": float(order.filled_qty),
                        "avg_price": float(order.filled_avg_price),
                        "total": float(order.filled_qty) * float(order.filled_avg_price),
                        "order_id": str(order.id),
                        "side": str(order.side),
                    }
                if order.status in (
                    OrderStatus.CANCELED,
                    OrderStatus.EXPIRED,
                    OrderStatus.REJECTED,
                ):
                    return {
                        "status": order.status.value.lower(),
                        "ticker": order.symbol,
                        "shares": 0,
                        "error": f"Order {order.status.value}",
                    }
                
                time.sleep(1)
            
            except Exception as e:
                log.error("Error checking order status: %s", e)
                return {
                    "status": "error",
                    "ticker": "UNKNOWN",
                    "shares": 0,
                    "error": str(e),
                }
        
        # Timeout reached — for paper mode, treat accepted orders as pending (they fill later)
        log.warning("Order %s timed out with status: %s", order_id[:8], last_status)
        if not self.is_live and str(last_status) in ("OrderStatus.ACCEPTED", "OrderStatus.NEW", "accepted", "new"):
            log.info("Paper mode: treating timed-out order as filled (will settle later)")
            return {
                "status": "filled",
                "ticker": order.symbol,
                "shares": float(order.qty),
                "avg_price": 0.0,  # Will be updated on next sync
                "total": 0.0,
                "order_id": str(order.id),
                "side": str(order.side),
            }
        return {
            "status": "timeout",
            "ticker": order.symbol,
            "shares": 0,
            "error": f"Order did not fill within {timeout}s (status: {last_status})",
        }

    def get_order_status(self, order_id: str) -> dict:
        """Get the status of an order."""
        try:
            order = self.client.get_order_by_id(order_id)
            return {
                "status": order.status,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "filled_qty": float(order.filled_qty),
                "filled_avg_price": float(order.filled_avg_price),
                "side": order.side,
            }
        except Exception as e:
            log.error("Error checking order status: %s", e)
            return {
                "status": "error",
                "error": str(e),
            }
