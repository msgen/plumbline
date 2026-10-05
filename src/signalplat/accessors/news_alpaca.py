"""Alpaca news (Benzinga content) as contract NewsItems."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd

from signalplat.contracts.types import NewsItem
from signalplat.utilities.http import JsonHttp
from signalplat.utilities.ratelimit import RateLimiter

# Provisional until E10 measures the real delay from publication to knowability.
DEFAULT_NEWS_LAG = timedelta(seconds=60)


class AlpacaNews:
    def __init__(
        self, http: JsonHttp, lag: timedelta = DEFAULT_NEWS_LAG, limiter: RateLimiter | None = None
    ) -> None:
        self._http, self._lag = http, lag
        self._limiter = limiter or RateLimiter(190, 60.0)

    def news(self, symbols: Sequence[str], start: datetime, end: datetime) -> list[NewsItem]:
        params = {
            "symbols": ",".join(symbols) or None, "start": start.isoformat(),
            "end": end.isoformat(), "limit": 50, "sort": "asc", "include_content": "false",
        }
        items: dict[str, NewsItem] = {}
        token = None
        while True:
            self._limiter.acquire()
            payload = self._http.get("/v1beta1/news", {**params, "page_token": token})
            for a in payload.get("news") or []:
                created = pd.Timestamp(a["created_at"]).tz_convert("UTC").to_pydatetime()
                items[str(a["id"])] = NewsItem(
                    id=str(a["id"]), created_at=created, available_at=created + self._lag,
                    headline=a.get("headline") or "", summary=a.get("summary") or "",
                    symbols=tuple(a.get("symbols") or ()),
                )
            token = payload.get("next_page_token")
            if not token:
                break
        return list(items.values())
