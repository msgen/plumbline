"""Ingestion manager: downloads, backfills and stores raw data. No formulas."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from signalplat.accessors.actions_alpaca import AlpacaActions
from signalplat.accessors.bars_alpaca import AlpacaBars
from signalplat.accessors.bars_parquet import ParquetBars
from signalplat.accessors.dataset_store import DatasetStore
from signalplat.accessors.filings_edgar import EdgarFilings
from signalplat.accessors.news_alpaca import AlpacaNews
from signalplat.accessors.reference_alpaca import AlpacaReference
from signalplat.contracts.types import Feed
from signalplat.engines.adjust import adjust_for_splits
from signalplat.engines.quality import sample_delisted
from signalplat.engines.universe import eligible_assets, select_universe
from signalplat.utilities.clock import Clock, SystemClock
from signalplat.utilities.config import load_config
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
        filings: EdgarFilings | None = None,
        actions: AlpacaActions | None = None,
    ) -> None:
        self._bars, self._reference, self._news = bars, reference, news
        self._filings, self._actions = filings, actions
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
        filings = None
        if env.get("SEC_USER_AGENT"):
            sec = {"User-Agent": env["SEC_USER_AGENT"]}
            filings = EdgarFilings(
                JsonHttp("https://www.sec.gov", sec), JsonHttp("https://data.sec.gov", sec)
            )
        return cls(
            AlpacaBars(data, clock), AlpacaReference(trading), AlpacaNews(data),
            ParquetBars(root), DatasetStore(root), clock, filings, AlpacaActions(data),
        )

    @property
    def skipped_symbols(self) -> dict[str, str]:
        """Symbols the data vendor refused (usually delisted or unknown), with the reason."""
        return dict(self._bars.skipped)

    @property
    def unmapped_filing_symbols(self) -> set[str]:
        """Symbols with no EDGAR CIK, so their filings are missing from the dataset."""
        return set(self._filings.unmapped) if self._filings else set()

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

    def ingest_splits(self, symbols: Sequence[str], start: datetime, end: datetime) -> int:
        if self._actions is None:
            raise RuntimeError("no corporate-actions accessor configured")
        total = 0
        for i in range(0, len(symbols), DAILY_CHUNK):
            chunk = list(symbols[i : i + DAILY_CHUNK])
            key = f"splits|{start:%F}|{end:%F}|{','.join(chunk)}"
            total += self._run(key, end, lambda c=chunk: self._store.write_splits(
                self._actions.splits(c, start, end)))
        return total

    def ingest_filings(self, symbols: Sequence[str], start: datetime, end: datetime) -> int:
        if self._filings is None:
            raise KeyError("missing required settings in .env: SEC_USER_AGENT")
        key = f"filings|{start:%F}|{end:%F}|{','.join(symbols)}"
        return self._run(key, end, lambda: self._store.write_filings(
            self._filings.filings(symbols, start, end)))

    def select_pilot(
        self, as_of: datetime, n: int, config_path: str | Path, lookback_days: int = 45
    ):
        """Fetch recent daily bars for every eligible stock, then keep the n most liquid.

        Uses only bars available before as_of. Needs the reference data to be ingested first.
        """
        assets = self._store.read_assets()
        if assets is None:
            raise RuntimeError("no asset list stored yet; run ingest --only reference first")
        candidates = eligible_assets(assets)
        start = as_of - timedelta(days=lookback_days)
        self.ingest_daily(candidates, start, as_of)
        self.ingest_splits(candidates, start, as_of)
        # raw bars, with only the splits effective by as_of applied: no look-ahead on price
        daily = adjust_for_splits(
            self._bars_store.daily_bars(candidates, start, as_of),
            self._store.read_splits(), as_of)
        return len(candidates), select_universe(daily, as_of, load_config(config_path), n)

    def ingest_delisted_sample(
        self, n: int, seed: int, start: datetime, end: datetime
    ) -> list[str]:
        """Fetch daily bars for a seeded sample of inactive symbols, for the E0 audit."""
        assets = self._store.read_assets()
        if assets is None:
            raise RuntimeError("no asset list stored yet; ingest the reference data first")
        sample = sample_delisted(assets, n, seed)
        self.ingest_daily(sample, start, end)
        if self._filings is not None:  # lets E0 measure how many delisted issuers have filings
            self.ingest_filings(sample, start, end)
        return sample

    def _run(self, key: str, chunk_end: datetime, fetch) -> int:
        if self._store.is_done(key):
            return 0
        rows = fetch()
        if chunk_end <= self._clock.now() - SAFE_LAG:
            self._store.mark_done(key, rows)
        log.info("chunk stored", extra={"fields": {"key": key[:80], "rows": rows}})
        return rows
