"""YouTube trader strategy extractor.

Fetches recent video transcripts from YouTube trader channels, then uses
GPT to extract their trading strategies and recommended tickers.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
log = logging.getLogger(__name__)


EXTRACT_PROMPT = """\
You are analysing transcripts from a YouTube trader's recent videos to extract
their trading strategy and any specific assets they are recommending or discussing.

Trader / Channel: {channel}

Recent video transcripts:
{transcripts}

Your task:
1. Identify the trader's general strategy (e.g. momentum, value, swing trading, crypto DCA etc.)
2. Extract any specific tickers, coins, or assets they are bullish or bearish on
3. Note any key signals they use (RSI, moving averages, news catalysts etc.)
4. Summarise the actionable trading ideas

IMPORTANT constraints on tickers you return:
- For stocks: any major US ticker (AAPL, TSLA, NVDA etc.)
- For crypto: ONLY BTC-USD, ETH-USD, DOGE-USD, LTC-USD, BCH-USD
- Ignore any other crypto tickers (SOL, ADA, XRP etc.) — they are not supported

Respond with this exact JSON:
{{
  "channel": "{channel}",
  "strategy_summary": "<2-3 sentence summary of the trader's strategy>",
  "bullish_tickers": ["TICK1", "TICK2"],
  "bearish_tickers": ["TICK3"],
  "key_signals": ["signal1", "signal2"],
  "confidence": <0.0-1.0 how clear and actionable their strategy is>
}}
"""


class YouTubeStrategyFetcher:
    """Fetches and analyses YouTube trader strategies."""

    def __init__(self):
        self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._yt_api_key = os.getenv("YOUTUBE_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tickers_from_channels(
        self,
        handles: list[str],
        max_videos_per_channel: int = 3,
    ) -> tuple[list[str], list[dict]]:
        """Main entry point.

        Fetches recent videos from each channel, extracts strategies,
        and returns a combined list of bullish tickers plus strategy summaries.

        Returns:
            (tickers, summaries) where tickers is a deduplicated list of
            bullish ticker symbols and summaries is a list of per-channel dicts.
        """
        all_tickers: list[str] = []
        summaries: list[dict] = []

        for handle in handles:
            handle = handle.strip().lstrip("@")
            log.info("Fetching strategy from YouTube channel: @%s", handle)
            try:
                result = self._process_channel(handle, max_videos_per_channel)
                if result:
                    summaries.append(result)
                    all_tickers.extend(result.get("bullish_tickers", []))
            except Exception as e:
                log.error("Failed to process channel @%s: %s", handle, e)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_tickers: list[str] = []
        for t in all_tickers:
            if t not in seen:
                seen.add(t)
                unique_tickers.append(t)

        return unique_tickers, summaries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process_channel(self, handle: str, max_videos: int) -> Optional[dict]:
        """Fetch videos from a channel and extract strategy."""
        video_ids = self._get_video_ids(handle, max_videos)
        if not video_ids:
            log.warning("No videos found for channel: @%s", handle)
            return None

        transcripts = []
        for vid_id in video_ids:
            text = self._get_transcript(vid_id)
            if text:
                transcripts.append(text[:3000])  # Cap per-video transcript length

        if not transcripts:
            log.warning("No transcripts available for @%s", handle)
            return None

        log.info("Analysing %d transcripts from @%s...", len(transcripts), handle)
        return self._extract_strategy(handle, transcripts)

    def _get_video_ids(self, handle: str, max_videos: int) -> list[str]:
        """Get recent video IDs from a channel using YouTube Data API."""
        if not self._yt_api_key:
            raise RuntimeError(
                "YOUTUBE_API_KEY is not set in your .env file. "
                "Get a free key at https://console.cloud.google.com/ → YouTube Data API v3."
            )

        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise RuntimeError(
                "Missing dependency: run `pip install google-api-python-client`"
            )

        youtube = build("youtube", "v3", developerKey=self._yt_api_key)

        # Resolve handle to channel ID
        channel_resp = youtube.channels().list(
            forHandle=handle,
            part="id,snippet",
        ).execute()

        items = channel_resp.get("items", [])
        if not items:
            # Try searching by name as fallback
            search_resp = youtube.search().list(
                q=handle,
                type="channel",
                part="id",
                maxResults=1,
            ).execute()
            search_items = search_resp.get("items", [])
            if not search_items:
                log.warning("Channel not found: @%s", handle)
                return []
            channel_id = search_items[0]["id"]["channelId"]
        else:
            channel_id = items[0]["id"]

        # Get recent uploads
        search_resp = youtube.search().list(
            channelId=channel_id,
            part="id",
            order="date",
            type="video",
            maxResults=max_videos,
        ).execute()

        return [
            item["id"]["videoId"]
            for item in search_resp.get("items", [])
            if item["id"].get("videoId")
        ]

    def _get_transcript(self, video_id: str) -> Optional[str]:
        """Fetch English transcript for a video using youtube-transcript-api."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
        except ImportError:
            raise RuntimeError(
                "Missing dependency: run `pip install youtube-transcript-api`"
            )

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            # Try English first, then auto-generated, then any
            try:
                transcript = transcript_list.find_transcript(["en"])
            except NoTranscriptFound:
                transcript = transcript_list.find_generated_transcript(["en"])

            chunks = transcript.fetch()
            return " ".join(chunk["text"] for chunk in chunks)

        except (NoTranscriptFound, TranscriptsDisabled):
            log.debug("No transcript for video %s", video_id)
            return None
        except Exception as e:
            log.debug("Transcript error for %s: %s", video_id, e)
            return None

    def _extract_strategy(self, channel: str, transcripts: list[str]) -> dict:
        """Use GPT to extract tickers and strategy from transcripts."""
        combined = "\n\n---\n\n".join(transcripts)
        prompt = EXTRACT_PROMPT.format(channel=channel, transcripts=combined)

        response = self._openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst extracting trading strategies from YouTube content.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            # Normalise tickers to uppercase
            data["bullish_tickers"] = [t.upper() for t in data.get("bullish_tickers", [])]
            data["bearish_tickers"] = [t.upper() for t in data.get("bearish_tickers", [])]
            return data
        except json.JSONDecodeError:
            log.error("Failed to parse strategy JSON for @%s", channel)
            return {"channel": channel, "bullish_tickers": [], "bearish_tickers": []}
