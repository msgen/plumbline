"""Alpaca corporate actions: forward and reverse splits.

Response field names (corporate_actions.forward_splits / reverse_splits with symbol, ex_date,
new_rate, old_rate) follow Alpaca's documentation as I know it and are unverified until the
first live run, so the parser fails loudly on a missing field.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from signalplat.utilities.http import JsonHttp
from signalplat.utilities.ratelimit import RateLimiter

SPLIT_COLUMNS = ["symbol", "ex_date", "ratio"]
_NEEDED = ("symbol", "ex_date", "new_rate", "old_rate")


class AlpacaActions:
    def __init__(self, http: JsonHttp, limiter: RateLimiter | None = None) -> None:
        self._http = http
        self._limiter = limiter or RateLimiter(190, 60.0)

    def splits(self, symbols: Sequence[str], start: datetime, end: datetime) -> pd.DataFrame:
        """ratio = new_rate / old_rate: 4.0 for a 4-for-1 split, 0.1 for a 1-for-10 reverse."""
        if not symbols:
            return pd.DataFrame(columns=SPLIT_COLUMNS)
        params = {
            "symbols": ",".join(symbols), "types": "forward_split,reverse_split",
            "start": start.date().isoformat(), "end": end.date().isoformat(), "limit": 1000,
        }
        rows: list[dict] = []
        token = None
        while True:
            self._limiter.acquire()
            payload = self._http.get("/v1/corporate-actions", {**params, "page_token": token})
            actions = payload.get("corporate_actions") or {}
            for kind in ("forward_splits", "reverse_splits"):
                for a in actions.get(kind) or []:
                    missing = [k for k in _NEEDED if k not in a]
                    if missing:
                        raise KeyError(f"corporate action lacks fields: {missing}")
                    rows.append({
                        "symbol": a["symbol"], "ex_date": a["ex_date"],
                        "ratio": float(a["new_rate"]) / float(a["old_rate"]),
                    })
            token = payload.get("next_page_token")
            if not token:
                break
        if not rows:
            return pd.DataFrame(columns=SPLIT_COLUMNS)
        df = pd.DataFrame(rows).drop_duplicates(["symbol", "ex_date"])
        return df.sort_values(["symbol", "ex_date"]).reset_index(drop=True)
