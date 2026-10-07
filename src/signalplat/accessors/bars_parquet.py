"""Local replay of stored bars (research path) and their writer."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from signalplat.contracts.types import MINUTE_SUMMARY_COLUMNS, Feed
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

    def minute_summary(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: Feed
    ) -> pd.DataFrame:
        """Per symbol-day facts about the stored minute bars, computed inside DuckDB.

        Minute rows are never loaded into Python, so this works at any size. Same table as
        `engines.quality.summarize_minutes`; a test keeps the two in step.
        """
        directory = self._root / "bars_raw_1m" / f"feed={feed.value}"
        if not symbols or not directory.exists() or not any(directory.rglob("part-*.parquet")):
            return pd.DataFrame(columns=MINUTE_SUMMARY_COLUMNS)
        marks = ",".join("?" for _ in symbols)
        sql = f"""
            WITH raw AS (
                SELECT DISTINCT symbol, timestamp, volume
                FROM read_parquet(?, union_by_name=true)
                WHERE symbol IN ({marks}) AND timestamp >= ? AND timestamp < ?
            ),
            local AS (
                SELECT symbol, timestamp, volume,
                       timezone('America/New_York', timestamp) AS lt
                FROM raw
            ),
            f AS (
                SELECT symbol, timestamp, volume, CAST(lt AS DATE) AS day,
                       hour(lt) * 60 + minute(lt) AS mod
                FROM local
            ),
            g AS (
                SELECT *, (mod >= 570 AND mod < 960) AS reg FROM f
            )
            SELECT symbol, day,
                   count(*) AS rows,
                   count(DISTINCT timestamp) AS timestamps,
                   count(DISTINCT CASE WHEN reg THEN mod END) AS reg_minutes,
                   min(CASE WHEN reg THEN mod END) AS reg_first,
                   max(CASE WHEN reg THEN mod END) AS reg_last,
                   sum(volume) AS vol_all,
                   coalesce(sum(CASE WHEN reg THEN volume END), 0) AS vol_regular,
                   count(*) FILTER (WHERE mod < 240 OR mod >= 1200) AS outside
            FROM g GROUP BY symbol, day
        """  # noqa: S608
        glob = str(directory / "**" / "part-*.parquet")
        con = duckdb.connect()
        try:
            out = con.execute(sql, [glob, *symbols, start, end]).df()
        finally:
            con.close()
        out[["reg_first", "reg_last"]] = out[["reg_first", "reg_last"]].astype(float)
        out["day"] = pd.to_datetime(out["day"]).dt.date  # plain dates, as the engines use
        return out[MINUTE_SUMMARY_COLUMNS]

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
