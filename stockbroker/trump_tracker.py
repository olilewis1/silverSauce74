"""Trump post & news tracker — extracts tradeable signals from Truth Social and news.

Sources (in order of priority):
1. Truth Social  — Trump's own posts via the public Mastodon-compatible API
2. NewsAPI.org   — Set NEWS_API_KEY in .env for 100 free requests/day
3. Google News RSS — Always-free fallback, no API key required

The AI analyses each new post/headline and decides:
- Which asset(s) are affected (ticker symbols)
- Whether the impact is bullish or bearish
- Confidence level and reasoning

Compatible with Alpaca for execution:
- Gold ETFs: GLD, IAU
- Energy: XOM, CVX
- Defense: RTX, LMT, NOC
- Crypto: BTC-USD, ETH-USD (Alpaca supported)
- Any US stock ticker
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
log = logging.getLogger(__name__)

TRUMP_TRUTH_ACCOUNT_ID = "107780257626128497"
SEEN_CACHE_FILE = Path("trump_seen_ids.json")

EXTRACT_PROMPT = """\
You are a financial analyst monitoring Donald Trump's social media posts and news headlines
to identify short-term trading opportunities.

Recent posts / headlines:
{content}

Your task:
1. Identify any assets, sectors, or topics mentioned that would move markets
2. Determine whether the sentiment is BULLISH or BEARISH for each asset
3. Extract specific ticker symbols to trade

STRICT rules on tickers you may return:
- Stocks: any major US ticker (GLD, IAU, XOM, CVX, RTX, LMT, TSLA, AAPL, etc.)
- Crypto: ONLY BTC-USD, ETH-USD, DOGE-USD, LTC-USD, BCH-USD
- Gold mentions → GLD or IAU
- Oil/energy mentions → XOM or CVX
- Defense/military mentions → RTX or LMT
- China trade war mentions → consider FXI (China ETF) as bearish or US manufacturers as bullish
- Do NOT return tickers you are not confident about
- Do NOT return tickers for assets Alpaca cannot trade (futures, forex, commodities)

Respond with this exact JSON:
{{
  "has_market_signal": true,
  "summary": "<1-2 sentence summary of what Trump said and why it matters>",
  "bullish_tickers": ["TICK1", "TICK2"],
  "bearish_tickers": ["TICK3"],
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation of why each ticker is affected>"
}}

If there is no meaningful market signal, set has_market_signal to false and leave ticker lists empty.
"""


class TrumpTracker:
    """Monitors Trump's Truth Social posts and news for tradeable signals."""

    def __init__(self) -> None:
        self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._news_api_key = os.getenv("NEWS_API_KEY", "").strip()
        self._seen: set[str] = self._load_seen()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tickers(self, max_items: int = 10) -> tuple[list[str], list[dict]]:
        """Fetch latest Trump posts/news, extract bullish tickers and summaries.

        Returns:
            (bullish_tickers, signal_summaries)
        """
        items = self._fetch_all(max_items)
        new_items = [i for i in items if i["id"] not in self._seen]

        if not new_items:
            log.info("No new Trump posts/news since last check.")
            return [], []

        log.info("Found %d new Trump signals to analyse.", len(new_items))

        result = self._extract_signals(new_items)

        for item in new_items:
            self._seen.add(item["id"])
        self._save_seen()

        if not result.get("has_market_signal"):
            log.info("No market-moving signal detected in latest Trump posts.")
            return [], []

        bullish = [t.upper() for t in result.get("bullish_tickers", [])]
        bearish = [t.upper() for t in result.get("bearish_tickers", [])]

        log.info(
            "Trump signal — bullish: %s | bearish: %s | confidence: %.0f%%",
            bullish or "none",
            bearish or "none",
            result.get("confidence", 0) * 100,
        )

        summary = {
            "source": "trump",
            "summary": result.get("summary", ""),
            "bullish_tickers": bullish,
            "bearish_tickers": bearish,
            "confidence": result.get("confidence", 0),
            "reasoning": result.get("reasoning", ""),
            "items_analysed": len(new_items),
        }

        return bullish, [summary]

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def _fetch_all(self, max_items: int) -> list[dict]:
        """Gather posts/headlines from all sources."""
        items: list[dict] = []

        ts = self._fetch_truth_social(max_items // 2 + 1)
        items.extend(ts)
        log.debug("Truth Social: %d posts", len(ts))

        news = self._fetch_news(max_items - len(ts))
        items.extend(news)
        log.debug("News: %d articles", len(news))

        return items[:max_items]

    def _fetch_truth_social(self, limit: int = 5) -> list[dict]:
        """Fetch recent Trump Truth Social posts via the public Mastodon API."""
        url = (
            f"https://truthsocial.com/api/v1/accounts/"
            f"{TRUMP_TRUTH_ACCOUNT_ID}/statuses?limit={limit}&exclude_replies=true"
        )
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; stockbroker/1.0)",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                posts = json.loads(resp.read())

            results = []
            for post in posts:
                content = post.get("content", "")
                content = _strip_html(content)
                if not content.strip():
                    continue
                post_id = f"ts_{post.get('id', hashlib.md5(content.encode()).hexdigest()[:8])}"
                results.append({
                    "id": post_id,
                    "source": "Truth Social",
                    "text": content,
                    "created_at": post.get("created_at", ""),
                })
            return results
        except Exception as e:
            log.warning("Truth Social fetch failed: %s", e)
            return []

    def _fetch_news(self, limit: int = 5) -> list[dict]:
        """Fetch Trump-related news. Uses NewsAPI if key set, else Google News RSS."""
        if self._news_api_key:
            return self._fetch_newsapi(limit)
        return self._fetch_google_news_rss(limit)

    def _fetch_newsapi(self, limit: int) -> list[dict]:
        """Fetch from NewsAPI.org (100 free requests/day with free tier)."""
        url = (
            "https://newsapi.org/v2/everything"
            "?q=trump+market+stock+gold+tariff+trade"
            "&sortBy=publishedAt"
            f"&pageSize={limit}"
            "&language=en"
            f"&apiKey={self._news_api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            articles = data.get("articles", [])
            return [
                {
                    "id": f"news_{hashlib.md5((a.get('url','') + a.get('title','')).encode()).hexdigest()[:10]}",
                    "source": a.get("source", {}).get("name", "News"),
                    "text": f"{a.get('title', '')}. {a.get('description', '')}",
                    "created_at": a.get("publishedAt", ""),
                }
                for a in articles if a.get("title")
            ]
        except Exception as e:
            log.warning("NewsAPI fetch failed: %s", e)
            return []

    def _fetch_google_news_rss(self, limit: int) -> list[dict]:
        """Free fallback — Google News RSS, no API key needed."""
        url = (
            "https://news.google.com/rss/search"
            "?q=trump+market+stocks+economy&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (compatible; stockbroker/1.0)"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")[:limit]
            results = []
            for item in items:
                title = item.findtext("title", "")
                desc  = item.findtext("description", "")
                link  = item.findtext("link", "")
                pub   = item.findtext("pubDate", "")
                text  = f"{title}. {_strip_html(desc)}"
                item_id = f"rss_{hashlib.md5(link.encode()).hexdigest()[:10]}"
                results.append({
                    "id": item_id,
                    "source": "Google News",
                    "text": text,
                    "created_at": pub,
                })
            return results
        except Exception as e:
            log.warning("Google News RSS fetch failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # AI extraction
    # ------------------------------------------------------------------

    def _extract_signals(self, items: list[dict]) -> dict:
        """Use GPT to extract tradeable signals from posts/headlines."""
        content_lines = [
            f"[{i['source']} @ {i['created_at'][:16]}] {i['text'][:400]}"
            for i in items
        ]
        content = "\n\n".join(content_lines)
        prompt = EXTRACT_PROMPT.format(content=content)

        response = self._openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst specialising in political event-driven trading.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.error("Failed to parse Trump signal JSON")
            return {"has_market_signal": False}

    # ------------------------------------------------------------------
    # Seen-ID cache  (avoids re-processing old posts every cycle)
    # ------------------------------------------------------------------

    def _load_seen(self) -> set[str]:
        if SEEN_CACHE_FILE.exists():
            try:
                return set(json.loads(SEEN_CACHE_FILE.read_text()))
            except Exception:
                pass
        return set()

    def _save_seen(self) -> None:
        try:
            ids = list(self._seen)[-500:]  # Keep last 500 to cap file size
            SEEN_CACHE_FILE.write_text(json.dumps(ids))
        except Exception as e:
            log.warning("Could not save seen IDs: %s", e)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    import re
    return re.sub(r"<[^>]+>", " ", text).strip()
