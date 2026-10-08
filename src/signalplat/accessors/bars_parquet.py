"""Local replay of stored bars (research path) and their writer."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from signalplat.contracts.types import MINUTE_SUMMARY_COLUMNS, Feed
from signalplat.utilities.clock import is_early_close
from signalplat.utilities.parquet import read_parts, write_part

COLUMNS = [
    "symbol", "timestamp", "open", "high", "low", "close",
    "volume", "vwap", "trade_count", "available_at",
]


def _early_close_days(start: datetime, end: datetime) -> list[date]:
    """Early-close dates in the window; never empty, so DuckDB can type the list."""
    days = [start.date() + timedelta(n) for n in range((end - start).days + 2)]
    return [d for d in days if is_early_close(d)] or [date(1900, 1, 1)]


_SUMMARY_SQL = """
    WITH raw AS (
        SELECT DISTINCT symbol, timestamp, volume, trade_count
        FROM read_parquet(?, union_by_name=true)
        WHERE symbol IN ({marks}) AND timestamp >= ? AND timestamp < ?
    ),
    local AS (
        SELECT symbol, timestamp, volume, trade_count,
               timezone('America/New_York', timestamp) AS lt
        FROM raw
    ),
    f AS (
        SELECT symbol, timestamp, volume, trade_count, CAST(lt AS DATE) AS day,
               hour(lt) * 60 + minute(lt) AS mod
        FROM local
    ),
    g AS (
        SELECT *, (mod >= 570
                AND mod < CASE WHEN list_contains(?, day) THEN 780 ELSE 960 END) AS reg FROM f
    )
    SELECT symbol, day,
           count(*) AS rows,
           count(DISTINCT timestamp) AS timestamps,
           count(DISTINCT CASE WHEN reg THEN mod END) AS reg_minutes,
           min(CASE WHEN reg THEN mod END) AS reg_first,
           max(CASE WHEN reg THEN mod END) AS reg_last,
           sum(volume) AS vol_all,
           coalesce(sum(CASE WHEN reg THEN volume END), 0) AS vol_regular,
           coalesce(sum(CASE WHEN reg THEN trade_count END), 0) AS trades_regular,
           count(*) FILTER (WHERE mod < 240 OR mod > 1200) AS outside,
           count(*) FILTER (WHERE mod = 1200) AS boundary
    FROM g GROUP BY symbol, day
"""


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

        Minute rows are never loaded into Python, and DuckDB sees one month directory at a
        time, so memory is bounded by a single month however long the history is. Same table
        as `engines.quality.summarize_minutes`; a test keeps the two in step.
        """
        directory = self._root / "bars_raw_1m" / f"feed={feed.value}"
        months = self._month_dirs(directory, start, end) if symbols else []
        if not months:
            return pd.DataFrame(columns=MINUTE_SUMMARY_COLUMNS)
        marks = ",".join("?" for _ in symbols)
        sql = _SUMMARY_SQL.replace("{marks}", marks)
        early = _early_close_days(start, end)
        parts = []
        con = duckdb.connect()
        try:
            con.execute("SET preserve_insertion_order = false")  # lets DuckDB stream and spill
            for month in months:
                glob = str(month / "part-*.parquet")
                parts.append(con.execute(sql, [glob, *symbols, start, end, early]).df())
        finally:
            con.close()
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame(columns=MINUTE_SUMMARY_COLUMNS)
        out = pd.concat(parts, ignore_index=True)
        # a New York day can straddle two UTC month directories (the 20:00 bar), so merge them
        out = out.groupby(["symbol", "day"], as_index=False).agg(
            rows=("rows", "sum"), timestamps=("timestamps", "sum"),
            reg_minutes=("reg_minutes", "sum"), reg_first=("reg_first", "min"),
            reg_last=("reg_last", "max"), vol_all=("vol_all", "sum"),
            vol_regular=("vol_regular", "sum"), trades_regular=("trades_regular", "sum"),
            outside=("outside", "sum"), boundary=("boundary", "sum"))
        out[["reg_first", "reg_last"]] = out[["reg_first", "reg_last"]].astype(float)
        out["day"] = pd.to_datetime(out["day"]).dt.date  # plain dates, as the engines use
        return out[MINUTE_SUMMARY_COLUMNS]

    @staticmethod
    def _month_dirs(directory: Path, start: datetime, end: datetime) -> list[Path]:
        """Month directories that can hold bars in [start, end), oldest first."""
        out = []
        for d in sorted(directory.glob("month=*")) if directory.exists() else []:
            if not any(d.glob("part-*.parquet")):
                continue
            first = datetime.strptime(d.name.removeprefix("month="), "%Y-%m").replace(tzinfo=UTC)
            last = datetime(first.year + (first.month == 12), first.month % 12 + 1, 1, tzinfo=UTC)
            if first < end and last > start:
                out.append(d)
        return out

    def opening_volume(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: Feed,
        minutes: Sequence[int], from_minute: int = 570,
    ) -> pd.DataFrame:
        """Volume and price*volume from `from_minute` (09:30) up to each cut-off, per symbol-day.

        `minutes` are minutes since midnight New York (575 = 09:35). Columns are v_HHMM and
        pv_HHMM (sum of bar vwap * volume), so VWAP is pv / v. Computed month by month in
        DuckDB, like `minute_summary`.
        """
        directory = self._root / "bars_raw_1m" / f"feed={feed.value}"
        months = self._month_dirs(directory, start, end) if symbols else []
        labels = [f"{int(m) // 60:02d}{int(m) % 60:02d}" for m in minutes]
        columns = ["symbol", "day", *[f"{k}_{lab}" for lab in labels for k in ("v", "pv")]]
        if not months:
            return pd.DataFrame(columns=columns)
        select = ",\n".join(
            f"coalesce(sum(CASE WHEN mod < {int(m)} THEN volume END), 0) AS v_{lab},"
            f" coalesce(sum(CASE WHEN mod < {int(m)} THEN vwap * volume END), 0) AS pv_{lab}"
            for m, lab in zip(minutes, labels, strict=True))
        marks = ",".join("?" for _ in symbols)
        sql = f"""
            WITH raw AS (
                SELECT DISTINCT symbol, timestamp, volume, vwap
                FROM read_parquet(?, union_by_name=true)
                WHERE symbol IN ({marks}) AND timestamp >= ? AND timestamp < ?
            ),
            f AS (
                SELECT symbol, volume, vwap, CAST(lt AS DATE) AS day,
                       hour(lt) * 60 + minute(lt) AS mod
                FROM (SELECT *, timezone('America/New_York', timestamp) AS lt FROM raw)
            )
            SELECT symbol, day, {select}
            FROM f WHERE mod >= {int(from_minute)} AND mod < {int(max(minutes))}
            GROUP BY symbol, day
        """  # noqa: S608
        parts = []
        con = duckdb.connect()
        try:
            con.execute("SET preserve_insertion_order = false")
            for month in months:
                glob = str(month / "part-*.parquet")
                parts.append(con.execute(sql, [glob, *symbols, start, end]).df())
        finally:
            con.close()
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame(columns=columns)
        out = pd.concat(parts, ignore_index=True)
        out["day"] = pd.to_datetime(out["day"]).dt.date
        return out.sort_values(["symbol", "day"]).reset_index(drop=True)[columns]

    def session_minute_bars(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: Feed
    ) -> pd.DataFrame:
        """Regular-session minute bars: symbol, day, mod, open..close, volume, vwap.

        Only 09:30 to the close (13:00 on early-close days); day is the New York date and mod
        the minute of the day. Read month by month, for research runs.
        """
        columns = ["symbol", "day", "mod", "open", "high", "low", "close", "volume", "vwap"]
        directory = self._root / "bars_raw_1m" / f"feed={feed.value}"
        months = self._month_dirs(directory, start, end) if symbols else []
        if not months:
            return pd.DataFrame(columns=columns)
        marks = ",".join("?" for _ in symbols)
        sql = f"""
            WITH raw AS (
                SELECT DISTINCT symbol, timestamp, open, high, low, close, volume, vwap
                FROM read_parquet(?, union_by_name=true)
                WHERE symbol IN ({marks}) AND timestamp >= ? AND timestamp < ?
            ),
            f AS (
                SELECT symbol, CAST(lt AS DATE) AS day, hour(lt) * 60 + minute(lt) AS mod,
                       open, high, low, close, volume, vwap
                FROM (SELECT *, timezone('America/New_York', timestamp) AS lt FROM raw)
            )
            SELECT * FROM f
            WHERE mod >= 570 AND mod < CASE WHEN list_contains(?, day) THEN 780 ELSE 960 END
        """  # noqa: S608
        early = _early_close_days(start, end)
        parts = []
        con = duckdb.connect()
        try:
            con.execute("SET preserve_insertion_order = false")
            for month in months:
                glob = str(month / "part-*.parquet")
                parts.append(con.execute(sql, [glob, *symbols, start, end, early]).df())
        finally:
            con.close()
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame(columns=columns)
        out = pd.concat(parts, ignore_index=True)
        out["day"] = pd.to_datetime(out["day"]).dt.date
        return out.sort_values(["symbol", "day", "mod"]).reset_index(drop=True)[columns]

    def outside_hours_examples(
        self, symbols: Sequence[str], start: datetime, end: datetime, feed: Feed, limit: int = 10
    ) -> list[tuple[str, str]]:
        """A few stored bars before 04:00 or after 20:00 New York, as (symbol, local time)."""
        directory = self._root / "bars_raw_1m" / f"feed={feed.value}"
        if not symbols or not directory.exists() or not any(directory.rglob("part-*.parquet")):
            return []
        marks = ",".join("?" for _ in symbols)
        sql = f"""
            SELECT symbol, strftime(timezone('America/New_York', timestamp), '%Y-%m-%d %H:%M')
            FROM read_parquet(?, union_by_name=true)
            WHERE symbol IN ({marks}) AND timestamp >= ? AND timestamp < ?
              AND (hour(timezone('America/New_York', timestamp)) < 4
                   OR hour(timezone('America/New_York', timestamp)) * 60
                      + minute(timezone('America/New_York', timestamp)) > 1200)
            ORDER BY 2 LIMIT {int(limit)}
        """  # noqa: S608
        glob = str(directory / "**" / "part-*.parquet")
        con = duckdb.connect()
        try:
            return [tuple(r) for r in con.execute(sql, [glob, *symbols, start, end]).fetchall()]
        finally:
            con.close()

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
