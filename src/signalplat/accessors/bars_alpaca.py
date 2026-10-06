"""Alpaca historical bars. Returns contract frames, never vendor shapes.

Bars are fetched raw (unadjusted). Splits are applied later, as of a given time, by the
adjust engine, so a split that happens after as_of can never change an earlier decision.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd

from signalplat.contracts.types import Feed
from signalplat.utilities.clock import Clock, clamp_sip_end
from signalplat.utilities.http import HttpError, JsonHttp
from signalplat.utilities.logging import get_logger
from signalplat.utilities.ratelimit import RateLimiter

BAR_COLUMNS = [
    "symbol", "timestamp", "open", "high", "low", "close",
    "volume", "vwap", "trade_count", "available_at",
]
log = get_logger("bars_alpaca")
_REJECTED = (400, 422)  # the vendor refused the request, typically over an unknown symbol
_TIMEFRAMES = {"1Min": timedelta(minutes=1), "1Day": timedelta(days=1)}


class AlpacaBars:
    """Free tier: SIP data must end 15+ minutes ago (clamped here); 200 calls a minute."""

    def __init__(self, http: JsonHttp, clock: Clock, limiter: RateLimiter | None = None) -> None:
        self._http, self._clock = http, clock
        self._limiter = limiter or RateLimiter(190, 60.0)
        self.skipped: dict[str, str] = {}  # symbols the vendor rejected, with the reason

    def minute_bars(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: Feed
    ) -> pd.DataFrame:
        return self._bars(symbols, start, end, feed, "1Min")

    def daily_bars(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: Feed = Feed.SIP
    ) -> pd.DataFrame:
        return self._bars(symbols, start, end, feed, "1Day")

    def _bars(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: Feed, timeframe: str
    ) -> pd.DataFrame:
        if feed is Feed.SIP:
            end = clamp_sip_end(end, self._clock.now())
        if not symbols or end <= start:
            return pd.DataFrame(columns=BAR_COLUMNS)
        try:
            return self._fetch(symbols, start, end, feed, timeframe)
        except HttpError as e:
            if e.status not in _REJECTED:
                raise
            if len(symbols) == 1:
                self.skipped[symbols[0]] = str(e)
                log.warning("symbol rejected", extra={"fields": {"symbol": symbols[0]}})
                return pd.DataFrame(columns=BAR_COLUMNS)
            mid = len(symbols) // 2  # bisect to isolate the offending symbols
            parts = [self._bars(symbols[:mid], start, end, feed, timeframe),
                     self._bars(symbols[mid:], start, end, feed, timeframe)]
            parts = [p for p in parts if not p.empty]
            return _concat(parts)

    def _fetch(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: Feed, timeframe: str
    ) -> pd.DataFrame:
        params = {
            "symbols": ",".join(symbols), "timeframe": timeframe,
            "start": start.isoformat(), "end": end.isoformat(),
            "limit": 10000, "adjustment": "raw", "feed": feed.value, "sort": "asc",
        }
        rows: list[dict] = []
        token = None
        while True:
            self._limiter.acquire()
            payload = self._http.get("/v2/stocks/bars", {**params, "page_token": token})
            for sym, bars in (payload.get("bars") or {}).items():
                rows.extend({"symbol": sym, **b} for b in bars)
            token = payload.get("next_page_token")
            if not token:
                break
        return _to_frame(rows, _TIMEFRAMES[timeframe])


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return pd.concat(parts).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _to_frame(rows: list[dict], width: timedelta) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=BAR_COLUMNS)
    df = pd.DataFrame(rows).rename(columns={
        "t": "timestamp", "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap", "n": "trade_count",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["available_at"] = df["timestamp"] + width  # a bar is knowable at its close
    return df[BAR_COLUMNS].sort_values(["symbol", "timestamp"]).reset_index(drop=True)
