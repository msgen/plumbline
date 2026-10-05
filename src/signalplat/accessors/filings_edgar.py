"""SEC EDGAR filings: 8-K item codes and offering forms, timestamped at SEC acceptance.

Field names follow EDGAR's submissions JSON as I know it (accessionNumber, form, items,
acceptanceDateTime, primaryDocument); the plan lists them as unverified until the first
live run, so the parser fails loudly on a missing field instead of guessing.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from signalplat.utilities.http import JsonHttp
from signalplat.utilities.ratelimit import RateLimiter

FILING_COLUMNS = [
    "symbol", "cik", "accession", "form", "items",
    "accepted_at", "available_at", "primary_document",
]
_KEEP = re.compile(r"^(8-K|S-1|S-3|424B\d)(/A)?$")
_NEEDED = ("accessionNumber", "form", "acceptanceDateTime")


class EdgarFilings:
    """`www` serves the ticker map, `data` serves submissions. SEC allows 10 requests a second."""

    def __init__(
        self, www: JsonHttp, data: JsonHttp, limiter: RateLimiter | None = None
    ) -> None:
        self._www, self._data = www, data
        self._limiter = limiter or RateLimiter(8, 1.0)
        self._ciks: dict[str, int] | None = None

    def _cik_map(self) -> dict[str, int]:
        if self._ciks is None:
            self._limiter.acquire()
            raw = self._www.get("/files/company_tickers.json")
            self._ciks = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
        return self._ciks

    def filings(self, symbols: Sequence[str], start: datetime, end: datetime) -> pd.DataFrame:
        ciks = self._cik_map()
        frames = []
        for sym in symbols:
            cik = ciks.get(sym.upper())
            if cik is None:  # delisted or renamed tickers are absent from the current map
                continue
            frames.append(self._one(sym, cik, start, end))
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=FILING_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _one(self, sym: str, cik: int, start: datetime, end: datetime) -> pd.DataFrame:
        self._limiter.acquire()
        sub = self._data.get(f"/submissions/CIK{cik:010d}.json")
        filings = sub.get("filings", {})
        blocks = [filings.get("recent", {})]
        for extra in filings.get("files", []):
            self._limiter.acquire()  # older history lives in additional pages
            blocks.append(self._data.get(f"/submissions/{extra['name']}"))
        rows = []
        for b in blocks:
            if not b or not b.get("accessionNumber"):
                continue
            missing = [k for k in _NEEDED if k not in b]
            if missing:
                raise KeyError(f"EDGAR submissions for CIK {cik} lack fields: {missing}")
            n = len(b["accessionNumber"])
            items = b.get("items") or [""] * n
            docs = b.get("primaryDocument") or [""] * n
            for i in range(n):
                if _KEEP.match(b["form"][i]):
                    rows.append({
                        "symbol": sym, "cik": cik, "accession": b["accessionNumber"][i],
                        "form": b["form"][i], "items": items[i],
                        "accepted_at": pd.Timestamp(b["acceptanceDateTime"][i]).tz_convert("UTC"),
                        "primary_document": docs[i],
                    })
        if not rows:
            return pd.DataFrame(columns=FILING_COLUMNS)
        df = pd.DataFrame(rows)
        df["available_at"] = df["accepted_at"]  # knowable at SEC acceptance
        df = df[(df["accepted_at"] >= start) & (df["accepted_at"] < end)]
        return df[FILING_COLUMNS].reset_index(drop=True)
