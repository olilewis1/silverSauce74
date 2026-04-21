"""Market data retrieval via yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yfinance as yf


@dataclass
class StockQuote:
    ticker: str
    price: float
    change_pct: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    sector: Optional[str]
    name: Optional[str]


def normalize_ticker(ticker: str) -> str:
    normalized = (ticker or "").upper().strip()
    if "-USD" in normalized or normalized.endswith("USD") or "/" in normalized:
        # Handle different crypto formats:
        # DOGE-USD -> DOGE-USD (keep)
        # DOGEUSD -> DOGE-USD (add dash)
        # DOGE/USD -> DOGE-USD (replace slash)
        if "-USD" in normalized:
            return normalized.replace("/", "-")  # Already has dash
        elif normalized.endswith("USD"):
            # Add dash before USD (DOGEUSD -> DOGE-USD)
            return normalized[:-3] + "-USD"
        return normalized.replace("/", "-")
    return normalized.replace(".", "-")


def get_quote(ticker: str) -> StockQuote:
    """Fetch a real-time quote for a single ticker."""
    normalized = normalize_ticker(ticker)
    t = yf.Ticker(normalized)
    info = t.info

    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or price
    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

    if price <= 0:
        raise ValueError(f"No quote data available for {normalized}")

    return StockQuote(
        ticker=normalized,
        price=price,
        change_pct=round(change_pct, 2),
        volume=info.get("volume") or info.get("regularMarketVolume") or 0,
        market_cap=info.get("marketCap"),
        pe_ratio=info.get("trailingPE"),
        dividend_yield=info.get("dividendYield"),
        fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
        fifty_two_week_low=info.get("fiftyTwoWeekLow"),
        sector=info.get("sector"),
        name=info.get("shortName") or info.get("longName"),
    )


def get_quotes(tickers: list[str]) -> dict[str, StockQuote]:
    """Fetch quotes for multiple tickers."""
    return {t: get_quote(t) for t in tickers}


def get_historical(ticker: str, period: str = "3mo", interval: str = "1d") -> dict:
    """Return historical OHLCV data as a dict for AI consumption."""
    normalized = normalize_ticker(ticker)
    t = yf.Ticker(normalized)
    df = t.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No historical data available for {normalized}")

    recent = df.tail(30)
    records = []
    for date, row in recent.iterrows():
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"]),
        })

    return {
        "ticker": normalized,
        "period": period,
        "avg_volume": int(df["Volume"].mean()),
        "volatility": round(df["Close"].pct_change().std() * (252 ** 0.5), 4),
        "price_change_pct": round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2),
        "recent_data": records,
    }


def get_stock_summary(ticker: str) -> dict:
    """Combine quote + historical data into a summary dict for the AI."""
    quote = get_quote(ticker)
    hist = get_historical(ticker)
    return {
        "ticker": quote.ticker,
        "name": quote.name,
        "price": quote.price,
        "change_pct": quote.change_pct,
        "volume": quote.volume,
        "market_cap": quote.market_cap,
        "pe_ratio": quote.pe_ratio,
        "dividend_yield": quote.dividend_yield,
        "52w_high": quote.fifty_two_week_high,
        "52w_low": quote.fifty_two_week_low,
        "sector": quote.sector,
        "volatility": hist.get("volatility"),
        "3mo_change_pct": hist.get("price_change_pct"),
        "avg_volume": hist.get("avg_volume"),
        "recent_prices": hist.get("recent_data", [])[-5:],
    }
