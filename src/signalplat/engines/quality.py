"""Data-quality checks behind experiment E0. Pure functions over frames: no I/O, no clock."""
from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import pandas as pd

from signalplat.utilities.clock import NY

MAJOR_EXCHANGES = ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")
REGULAR_MINUTES = 390
EXTENDED_OPEN, EXTENDED_CLOSE = 4 * 60, 20 * 60  # 04:00-20:00 New York
REGULAR_OPEN, REGULAR_CLOSE = 9 * 60 + 30, 16 * 60


def sample_delisted(
    assets: pd.DataFrame, n: int, seed: int, exchanges: Sequence[str] = MAJOR_EXCHANGES
) -> list[str]:
    """Seeded sample of inactive symbols on major exchanges (OTC names excluded)."""
    inactive = assets[(assets["status"] == "inactive") & assets["exchange"].isin(exchanges)]
    symbols = sorted(inactive["symbol"].unique())
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


def duplicate_count(bars: pd.DataFrame) -> int:
    return int(bars.duplicated(["symbol", "timestamp"]).sum()) if len(bars) else 0


def _local(bars: pd.DataFrame) -> pd.DataFrame:
    local = bars["timestamp"].dt.tz_convert(NY)
    out = bars.assign(day=local.dt.date, mod=local.dt.hour * 60 + local.dt.minute)
    return out


def outside_extended_hours(minute: pd.DataFrame) -> int:
    if minute.empty:
        return 0
    m = _local(minute)["mod"]
    return int(((m < EXTENDED_OPEN) | (m >= EXTENDED_CLOSE)).sum())


def missing_regular_minutes(minute: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    """Share of regular-session minutes missing for symbol-days that have a daily bar.

    The expected session each day is the market-wide first to last regular minute seen, so
    half days are not counted as missing. A day with no minute data at all expects 390.
    """
    if daily.empty:
        return {"symbol_days": 0, "expected": 0, "missing": 0, "fraction": None, "worst": []}
    d = _local(daily)[["symbol", "day"]].drop_duplicates()
    m = _local(minute) if len(minute) else minute.assign(day=[], mod=[])
    reg = m[(m["mod"] >= REGULAR_OPEN) & (m["mod"] < REGULAR_CLOSE)]
    span = reg.groupby("day")["mod"].agg(["min", "max"])
    span["expected"] = span["max"] - span["min"] + 1
    obs = reg.groupby(["symbol", "day"])["mod"].nunique().rename("observed")
    d = d.join(span["expected"], on="day").join(obs, on=["symbol", "day"])
    d["expected"] = d["expected"].fillna(REGULAR_MINUTES)
    d["observed"] = d["observed"].fillna(0).clip(upper=d["expected"])
    d["missing"] = d["expected"] - d["observed"]
    expected, missing = float(d["expected"].sum()), float(d["missing"].sum())
    worst = d.sort_values("missing", ascending=False).head(10)
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
    minute: pd.DataFrame, daily: pd.DataFrame, tolerance: float = 0.02
) -> dict[str, Any]:
    """Compare daily volume with the minute total, both all-hours and regular-hours only.

    Which convention Alpaca's daily bar follows is unverified, so the closer one is used.
    """
    if daily.empty or minute.empty:
        return {"symbol_days": 0, "median_rel_diff": None, "within_tolerance": None}
    m = _local(minute)
    all_hours = m.groupby(["symbol", "day"])["volume"].sum().rename("all_hours")
    regular = (
        m[(m["mod"] >= REGULAR_OPEN) & (m["mod"] < REGULAR_CLOSE)]
        .groupby(["symbol", "day"])["volume"].sum().rename("regular")
    )
    d = _local(daily)[["symbol", "day", "volume"]].join(all_hours, on=["symbol", "day"])
    d = d.join(regular, on=["symbol", "day"]).dropna(subset=["all_hours"])
    d = d[d["volume"] > 0]
    if d.empty:
        return {"symbol_days": 0, "median_rel_diff": None, "within_tolerance": None}
    best: dict[str, Any] | None = None
    for name in ("all_hours", "regular"):
        rel = (d["volume"] - d[name].fillna(0)).abs() / d["volume"]
        cand = {
            "symbol_days": len(d), "convention": name,
            "median_rel_diff": float(rel.median()),
            "within_tolerance": float((rel <= tolerance).mean()),
        }
        if best is None or cand["median_rel_diff"] < best["median_rel_diff"]:
            best = cand
    return best  # type: ignore[return-value]


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
