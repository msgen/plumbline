"""Ingestion manager: downloads, backfills and stores raw data. No formulas."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from signalplat.accessors.bars_alpaca import AlpacaBars
from signalplat.accessors.bars_parquet import ParquetBars
from signalplat.accessors.dataset_store import DatasetStore
from signalplat.accessors.news_alpaca import AlpacaNews
from signalplat.accessors.reference_alpaca import AlpacaReference
from signalplat.contracts.types import Feed
from signalplat.utilities.clock import Clock, SystemClock
from signalplat.utilities.env import load_env, require
from signalplat.utilities.http import JsonHttp
from signalplat.utilities.logging import get_logger

log = get_logger("ingestion")
DAILY_CHUNK = 100
MINUTE_CHUNK = 5
SAFE_LAG = timedelta(minutes=16)  # chunks ending later may be clamped, so are not marked done


def month_starts(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into calendar-month windows (UTC)."""
    out, cur = [], start
    while cur < end:
        nxt = datetime(cur.year + (cur.month == 12), cur.month % 12 + 1, 1, tzinfo=UTC)
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out


class IngestionManager:
    def __init__(
        self,
        bars: AlpacaBars,
        reference: AlpacaReference,
        news: AlpacaNews,
        bars_store: ParquetBars,
        store: DatasetStore,
        clock: Clock,
    ) -> None:
        self._bars, self._reference, self._news = bars, reference, news
        self._bars_store, self._store, self._clock = bars_store, store, clock

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> IngestionManager:
        env = env if env is not None else load_env()
        require(env, "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")
        headers = {
            "APCA-API-KEY-ID": env["ALPACA_API_KEY_ID"],
            "APCA-API-SECRET-KEY": env["ALPACA_API_SECRET_KEY"],
        }
        data = JsonHttp(env.get("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"), headers)
        trading = JsonHttp(
            env.get("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets"), headers
        )
        root = Path(env.get("DATA_DIR", "./data"))
        clock = SystemClock()
        return cls(
            AlpacaBars(data, clock), AlpacaReference(trading), AlpacaNews(data),
            ParquetBars(root), DatasetStore(root), clock,
        )

    def ingest_reference(self) -> int:
        df = self._reference.assets()
        self._store.write_assets(df)
        log.info("assets stored", extra={"fields": {"rows": len(df)}})
        return len(df)

    def ingest_daily(self, symbols: Sequence[str], start: datetime, end: datetime) -> int:
        total = 0
        for i in range(0, len(symbols), DAILY_CHUNK):
            chunk = list(symbols[i : i + DAILY_CHUNK])
            key = f"1d|{start:%F}|{end:%F}|{','.join(chunk)}"
            total += self._run(key, end, lambda c=chunk: self._bars_store.write_daily(
                self._bars.daily_bars(c, start, end)))
        return total

    def ingest_minute(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: Feed
    ) -> int:
        total = 0
        for lo, hi in month_starts(start, end):
            for i in range(0, len(symbols), MINUTE_CHUNK):
                chunk = list(symbols[i : i + MINUTE_CHUNK])
                key = f"1m|{feed.value}|{lo:%F}|{hi:%F}|{','.join(chunk)}"
                total += self._run(key, hi, lambda c=chunk, a=lo, b=hi: (
                    self._bars_store.write_minute(self._bars.minute_bars(c, a, b, feed), feed)))
        return total

    def ingest_news(self, symbols: Sequence[str], start: datetime, end: datetime) -> int:
        total = 0
        for lo, hi in month_starts(start, end):
            key = f"news|{lo:%F}|{hi:%F}|{','.join(symbols)}"
            total += self._run(key, hi, lambda a=lo, b=hi: self._store.write_news(
                self._news.news(symbols, a, b)))
        return total

    def _run(self, key: str, chunk_end: datetime, fetch) -> int:
        if self._store.is_done(key):
            return 0
        rows = fetch()
        if chunk_end <= self._clock.now() - SAFE_LAG:
            self._store.mark_done(key, rows)
        log.info("chunk stored", extra={"fields": {"key": key[:80], "rows": rows}})
        return rows
