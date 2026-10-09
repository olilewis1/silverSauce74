#!/usr/bin/env python3
"""Lightweight position monitor — no OpenAI, no cost.

Reads portfolio_state.json, fetches live prices from yfinance (free),
and sends a push notification if any position needs attention.

Run this on a schedule (cron, launchd) — e.g. every 15 minutes.

Setup:
  Add NTFY_TOPIC=your-topic to .env  (see .env.example)
  cron: */15 * * * * cd /path/to/project && venv/bin/python3 monitor.py
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STATE_FILE = Path("portfolio_state.json")

# Thresholds — override via env vars or edit here
STOP_LOSS_PCT   = float(os.getenv("MONITOR_STOP_LOSS",   "8"))   / 100
TAKE_PROFIT_PCT = float(os.getenv("MONITOR_TAKE_PROFIT", "25"))  / 100
WARN_BUFFER     = float(os.getenv("MONITOR_WARN_BUFFER", "2"))   / 100  # warn this % before limit


def get_price(ticker: str) -> float | None:
    """Fetch latest price from yfinance (free, no API key)."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        return float(info.last_price)
    except Exception:
        return None


def ntfy(title: str, message: str, priority: str = "default") -> None:
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode(),
            headers={"Title": title, "Priority": priority, "Tags": "chart_with_upwards_trend"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"ntfy failed: {e}")


def main() -> None:
    if not STATE_FILE.exists():
        print("No portfolio_state.json found — nothing to monitor.")
        return

    state = json.loads(STATE_FILE.read_text())
    positions = state.get("positions", {})

    if not positions:
        print("No open positions.")
        return

    alerts: list[tuple[str, str, str]] = []  # (title, message, priority)

    for ticker, pos in positions.items():
        avg_cost = pos.get("avg_cost", 0)
        shares   = pos.get("shares", 0)
        if avg_cost <= 0 or shares <= 0:
            continue

        price = get_price(ticker)
        if price is None:
            print(f"Could not fetch price for {ticker}")
            continue

        pnl_pct = (price - avg_cost) / avg_cost
        value   = shares * price

        print(f"{ticker}: {pnl_pct:+.1%}  (avg ${avg_cost:.2f} → now ${price:.2f}, {shares:.4g} shares = ${value:.2f})")

        # Stop-loss check
        if pnl_pct <= -STOP_LOSS_PCT:
            alerts.append((
                f"🔴 SELL NOW — {ticker} stop-loss hit",
                f"{ticker} is down {pnl_pct:.1%} (limit: {STOP_LOSS_PCT:.0%}). "
                f"Open your broker and sell {shares:.4g} shares.",
                "urgent",
            ))
        elif pnl_pct <= -(STOP_LOSS_PCT - WARN_BUFFER):
            alerts.append((
                f"⚠️ Warning — {ticker} nearing stop-loss",
                f"{ticker} is down {pnl_pct:.1%}, approaching your {STOP_LOSS_PCT:.0%} stop-loss.",
                "high",
            ))

        # Take-profit check
        elif pnl_pct >= TAKE_PROFIT_PCT:
            alerts.append((
                f"🟢 SELL NOW — {ticker} take-profit hit",
                f"{ticker} is up {pnl_pct:.1%} (target: {TAKE_PROFIT_PCT:.0%}). "
                f"Open your broker and consider selling {shares:.4g} shares.",
                "urgent",
            ))
        elif pnl_pct >= (TAKE_PROFIT_PCT - WARN_BUFFER):
            alerts.append((
                f"📈 {ticker} nearing take-profit",
                f"{ticker} is up {pnl_pct:.1%}, approaching your {TAKE_PROFIT_PCT:.0%} target.",
                "high",
            ))

    if alerts:
        for title, message, priority in alerts:
            print(f"\nALERT: {title}\n  {message}")
            ntfy(title, message, priority)
    else:
        print("\nAll positions within range — no action needed.")


if __name__ == "__main__":
    main()
