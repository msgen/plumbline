"""Local replay of stored bars (research path) and their writer."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from signalplat.contracts.types import Feed
from signalplat.utilities.parquet import read_parts, write_part

COLUMNS = [
    "symbol", "timestamp", "open", "high", "low", "close",
    "volume", "vwap", "trade_count", "available_at",
]


class ParquetBars:
    """Raw bars. Layout: bars_raw_1d/year=YYYY/ and bars_raw_1m/feed=F/month=YYYY-MM/ of parts."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def write_daily(self, df: pd.DataFrame) -> int:
        n = 0
        for year, g in df.groupby(df["timestamp"].dt.year):
            write_part(g, self._root / "bars_raw_1d" / f"year={year}")
            n += len(g)
        return n

    def write_minute(self, df: pd.DataFrame, feed: Feed) -> int:
        n = 0
        for month, g in df.groupby(df["timestamp"].dt.strftime("%Y-%m")):
            write_part(g, self._root / "bars_raw_1m" / f"feed={feed.value}" / f"month={month}")
            n += len(g)
        return n

    def minute_bars(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: Feed
    ) -> pd.DataFrame:
        return self._read(self._root / "bars_raw_1m" / f"feed={feed.value}", symbols, start, end)

    def daily_bars(self, symbols: Sequence[str], start: datetime, end: datetime) -> pd.DataFrame:
        return self._read(self._root / "bars_raw_1d", symbols, start, end)

    @staticmethod
    def _read(
        directory: Path, symbols: Sequence[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame(columns=COLUMNS)
        marks = ",".join("?" for _ in symbols)
        df = read_parts(
            directory,
            f"symbol IN ({marks}) AND timestamp >= ? AND timestamp < ?",
            [*symbols, start, end],
        )
        if df is None:
            return pd.DataFrame(columns=COLUMNS)
        return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
