"""Data-quality checks behind experiment E0. Pure functions over frames: no I/O, no clock."""
from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

from signalplat.contracts.types import MINUTE_SUMMARY_COLUMNS
from signalplat.utilities.clock import NY

_TICKER = re.compile(r"^[A-Z]{1,5}$")  # drops escrow, CVR and CUSIP-style placeholders
MAJOR_EXCHANGES = ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")
REGULAR_MINUTES = 390
EXTENDED_OPEN, EXTENDED_CLOSE = 4 * 60, 20 * 60  # 04:00-20:00 New York
REGULAR_OPEN, REGULAR_CLOSE = 9 * 60 + 30, 16 * 60


def sample_delisted(
    assets: pd.DataFrame, n: int, seed: int, exchanges: Sequence[str] = MAJOR_EXCHANGES
) -> list[str]:
    """Seeded sample of inactive, ticker-like symbols on major exchanges (OTC excluded)."""
    inactive = assets[(assets["status"] == "inactive") & assets["exchange"].isin(exchanges)]
    symbols = sorted(s for s in inactive["symbol"].unique() if _TICKER.match(s))
    return sorted(random.Random(seed).sample(symbols, min(n, len(symbols))))


def delisted_coverage(sample: Sequence[str], daily: pd.DataFrame) -> dict[str, Any]:
    present = set(daily["symbol"].unique()) if len(daily) else set()
    missing = sorted(set(sample) - present)
    total = len(sample)
    return {
        "sampled": total,
        "present": total - len(missing),
        "fraction": (total - len(missing)) / total if total else None,
        "missing": missing,
    }


def _local(bars: pd.DataFrame) -> pd.DataFrame:
    local = bars["timestamp"].dt.tz_convert(NY)
    return bars.assign(day=local.dt.date, mod=local.dt.hour * 60 + local.dt.minute)


def summarize_minutes(minute: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol-day of minute-bar facts: what the E0 checks need, and nothing more.

    This is the reference implementation. The production path computes the same table inside
    DuckDB (`ParquetBars.minute_summary`) so minute rows never have to fit in memory.
    """
    if minute.empty:
        return pd.DataFrame(columns=MINUTE_SUMMARY_COLUMNS)
    m = _local(minute)
    reg = (m["mod"] >= REGULAR_OPEN) & (m["mod"] < REGULAR_CLOSE)
    m = m.assign(
        reg_mod=m["mod"].where(reg),
        vol_regular=m["volume"].where(reg, 0.0),
        outside=((m["mod"] < EXTENDED_OPEN) | (m["mod"] >= EXTENDED_CLOSE)).astype(int),
    )
    g = m.groupby(["symbol", "day"])
    out = pd.DataFrame({
        "rows": g.size(), "timestamps": g["timestamp"].nunique(),
        "reg_minutes": g["reg_mod"].nunique(), "reg_first": g["reg_mod"].min(),
        "reg_last": g["reg_mod"].max(), "vol_all": g["volume"].sum(),
        "vol_regular": g["vol_regular"].sum(), "outside": g["outside"].sum(),
    }).reset_index()
    return out[MINUTE_SUMMARY_COLUMNS]


def duplicate_count(bars: pd.DataFrame) -> int:
    return int(bars.duplicated(["symbol", "timestamp"]).sum()) if len(bars) else 0


def duplicate_minute_rows(summary: pd.DataFrame) -> int:
    return int((summary["rows"] - summary["timestamps"]).sum()) if len(summary) else 0


def outside_extended_hours(summary: pd.DataFrame) -> int:
    return int(summary["outside"].sum()) if len(summary) else 0


def missing_regular_minutes(summary: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    """Share of regular-session minutes missing for symbol-days that have a daily bar.

    The expected session each day is the market-wide first to last regular minute seen, so
    half days are not counted as missing. A day with no minute data at all expects 390.
    """
    if daily.empty:
        return {"symbol_days": 0, "expected": 0, "missing": 0, "fraction": None, "worst": []}
    d = _local(daily)[["symbol", "day"]].drop_duplicates()
    if len(summary):
        span = summary.groupby("day").agg(first=("reg_first", "min"), last=("reg_last", "max"))
        span["expected"] = span["last"] - span["first"] + 1
        obs = summary.set_index(["symbol", "day"])["reg_minutes"].rename("observed")
        d = d.join(span["expected"], on="day").join(obs, on=["symbol", "day"])
    else:
        d["expected"], d["observed"] = float("nan"), float("nan")
    d["expected"] = d["expected"].fillna(REGULAR_MINUTES)
    d["observed"] = d["observed"].fillna(0).clip(upper=d["expected"])
    d["missing"] = d["expected"] - d["observed"]
    expected, missing = float(d["expected"].sum()), float(d["missing"].sum())
    worst = d[d["missing"] > 0].sort_values("missing", ascending=False).head(10)
    return {
        "symbol_days": len(d), "expected": int(expected), "missing": int(missing),
        "fraction": missing / expected if expected else None,
        "worst": [(r.symbol, str(r.day), int(r.missing)) for r in worst.itertuples()],
    }


def price_jumps(daily: pd.DataFrame, threshold: float = 0.40) -> list[tuple[str, str, float]]:
    """Close-to-close moves beyond threshold in split-adjusted data: each needs a human look."""
    if daily.empty:
        return []
    d = daily.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    d["ret"] = d.groupby("symbol")["close"].pct_change()
    d["day"] = d["timestamp"].dt.tz_convert(NY).dt.date
    hit = d[d["ret"].abs() > threshold]
    return [(r.symbol, str(r.day), float(r.ret)) for r in hit.itertuples()]


def volume_consistency(
    summary: pd.DataFrame, daily: pd.DataFrame, tolerance: float = 0.02
) -> dict[str, Any]:
    """Compare daily volume with the minute total, both all-hours and regular-hours only.

    Which convention Alpaca's daily bar follows is unverified, so the closer one is used.
    """
    empty = {"symbol_days": 0, "median_rel_diff": None, "within_tolerance": None}
    if daily.empty or summary.empty:
        return empty
    d = _local(daily)[["symbol", "day", "volume"]].join(
        summary.set_index(["symbol", "day"])[["vol_all", "vol_regular"]], on=["symbol", "day"]
    ).dropna(subset=["vol_all"])
    d = d[d["volume"] > 0]
    if d.empty:
        return empty
    best: dict[str, Any] | None = None
    for name in ("vol_all", "vol_regular"):
        rel = (d["volume"] - d[name].fillna(0)).abs() / d["volume"]
        cand = {
            "symbol_days": len(d),
            "convention": {"vol_all": "all_hours", "vol_regular": "regular"}[name],
            "median_rel_diff": float(rel.median()),
            "within_tolerance": float((rel <= tolerance).mean()),
        }
        if best is None or cand["median_rel_diff"] < best["median_rel_diff"]:
            best = cand
    return best  # type: ignore[return-value]


def filings_coverage(
    universe: Sequence[str], sample: Sequence[str], filings: pd.DataFrame | None
) -> dict[str, Any]:
    """Share of symbols with at least one stored filing, for current names and delisted ones.

    A big gap between the two means catalyst data has survivorship bias.
    """
    have = set(filings["symbol"].unique()) if filings is not None and len(filings) else set()

    def share(names: Sequence[str]) -> dict[str, Any]:
        n = len(names)
        k = len(set(names) & have)
        return {"with_filings": k, "total": n, "fraction": k / n if n else None}

    return {"universe": share(universe), "delisted_sample": share(sample)}


def news_coverage(
    news: pd.DataFrame | None, symbols: Sequence[str], days: int
) -> dict[str, Any]:
    """Articles per calendar day and the share of symbols with any article."""
    if news is None or news.empty:
        return {"articles": 0, "per_day": 0.0, "symbols_with_news": 0, "symbols": len(symbols)}
    tagged = set(news["symbols"].explode().dropna().unique())
    return {
        "articles": int(news["id"].nunique()),
        "per_day": float(news["id"].nunique() / max(days, 1)),
        "symbols_with_news": len(set(symbols) & tagged),
        "symbols": len(symbols),
    }


def evaluate_e0(metrics: dict[str, Any], gate: dict[str, float]) -> dict[str, dict[str, Any]]:
    """Turn raw metrics into pass/fail/review lines. Failing a gate never loosens it."""

    def line(value, threshold, ok, op) -> dict[str, Any]:
        status = "n/a" if value is None else ("pass" if ok(value) else "fail")
        return {"value": value, "threshold": threshold, "op": op, "status": status}

    jumps = metrics["price_jumps"]
    return {
        "delisted_present": line(metrics["delisted"]["fraction"], gate["delisted_present_min"],
                                 lambda v: v >= gate["delisted_present_min"], ">="),
        "missing_regular_minutes": line(
            metrics["missing_minutes"]["fraction"], gate["missing_regular_minutes_max"],
            lambda v: v <= gate["missing_regular_minutes_max"], "<="),
        "daily_vs_minute_volume": line(
            metrics["volume"]["median_rel_diff"], gate["daily_vs_minute_volume_tol"],
            lambda v: v <= gate["daily_vs_minute_volume_tol"], "<="),
        "duplicates": line(metrics["duplicates"], 0, lambda v: v == 0, "=="),
        "bars_outside_extended_hours": line(
            metrics["outside_hours"], 0, lambda v: v == 0, "=="),
        # Large moves can be real (earnings, takeovers); they are listed, not auto-failed.
        "price_jumps": {"value": len(jumps), "threshold": 0, "op": "==",
                        "status": "pass" if not jumps else "review"},
    }
