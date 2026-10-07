"""Data-quality checks behind experiment E0. Pure functions over frames: no I/O, no clock."""
from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

import pandas as pd

from signalplat.contracts.types import MINUTE_SUMMARY_COLUMNS
from signalplat.utilities.clock import NY, is_early_close, regular_close_minute

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
    close = m["day"].map(regular_close_minute)  # 13:00 on early-close days
    reg = (m["mod"] >= REGULAR_OPEN) & (m["mod"] < close)
    m = m.assign(
        reg_mod=m["mod"].where(reg),
        vol_regular=m["volume"].where(reg, 0.0),
        trades_regular=(m["trade_count"].where(reg, 0.0) if "trade_count" in m else 0.0),
        outside=((m["mod"] < EXTENDED_OPEN) | (m["mod"] > EXTENDED_CLOSE)).astype(int),
        boundary=(m["mod"] == EXTENDED_CLOSE).astype(int),
    )
    g = m.groupby(["symbol", "day"])
    out = pd.DataFrame({
        "rows": g.size(), "timestamps": g["timestamp"].nunique(),
        "reg_minutes": g["reg_mod"].nunique(), "reg_first": g["reg_mod"].min(),
        "reg_last": g["reg_mod"].max(), "vol_all": g["volume"].sum(),
        "vol_regular": g["vol_regular"].sum(), "trades_regular": g["trades_regular"].sum(),
        "outside": g["outside"].sum(), "boundary": g["boundary"].sum(),
    }).reset_index()
    return out[MINUTE_SUMMARY_COLUMNS]


def minute_gaps(minutes_of_day: Sequence[int], top: int = 5) -> list[tuple[str, int]]:
    """Longest runs of absent minutes between the first and last bar: (start HH:MM, length)."""
    ms = sorted(set(minutes_of_day))
    gaps = [(a + 1, b - a - 1) for a, b in zip(ms, ms[1:], strict=False) if b - a > 1]
    gaps.sort(key=lambda g: -g[1])
    return [(f"{start // 60:02d}:{start % 60:02d}", n) for start, n in gaps[:top]]


def duplicate_count(bars: pd.DataFrame) -> int:
    return int(bars.duplicated(["symbol", "timestamp"]).sum()) if len(bars) else 0


def duplicate_minute_rows(summary: pd.DataFrame) -> int:
    return int((summary["rows"] - summary["timestamps"]).sum()) if len(summary) else 0


def outside_extended_hours(summary: pd.DataFrame) -> int:
    """Bars before 04:00 or after 20:00 New York. The 20:00 minute itself is the session's
    closing boundary, counted separately by `boundary_bars`."""
    return int(summary["outside"].sum()) if len(summary) else 0


def boundary_bars(summary: pd.DataFrame) -> int:
    return int(summary["boundary"].sum()) if len(summary) else 0


def missing_regular_minutes(
    summary: pd.DataFrame, daily: pd.DataFrame, members: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Share of regular-session minutes missing for symbol-days that have a daily bar.

    With `members` (symbol, day, member), `fraction` covers universe days only, which is what
    the gate is about: a thin stock legitimately has minutes with no trades. `fraction_all`
    always covers every day.

    The expected session each day is the market-wide first to last regular minute seen, so
    half days are not counted as missing. A day with no minute data at all expects 390.
    """
    if daily.empty:
        return {"symbol_days": 0, "expected": 0, "missing": 0, "fraction": None, "worst": []}
    d = _local(daily)[["symbol", "day", "volume", "close"]].drop_duplicates(["symbol", "day"])
    d = d.rename(columns={"close": "price"})
    if len(summary):
        span = summary.groupby("day").agg(first=("reg_first", "min"), last=("reg_last", "max"))
        span["expected"] = span["last"] - span["first"] + 1
        obs = summary.set_index(["symbol", "day"])["reg_minutes"].rename("observed")
        d = d.join(span["expected"], on="day").join(obs, on=["symbol", "day"])
    else:
        d["expected"], d["observed"] = float("nan"), float("nan")
    fallback = d["day"].map(lambda x: 210 if is_early_close(x) else REGULAR_MINUTES)
    d["expected"] = d["expected"].fillna(fallback)
    d["observed"] = d["observed"].fillna(0).clip(upper=d["expected"])
    d["missing"] = d["expected"] - d["observed"]
    if len(summary):
        cols = ["trades_regular", "vol_all", "vol_regular"]
        d = d.join(summary.set_index(["symbol", "day"])[cols], on=["symbol", "day"])
    else:
        d["trades_regular"], d["vol_all"], d["vol_regular"] = 0.0, float("nan"), float("nan")
    all_expected, all_missing = float(d["expected"].sum()), float(d["missing"].sum())
    if members is not None:
        keep = members.loc[members["member"], ["symbol", "day"]]
        d = d.merge(keep, on=["symbol", "day"], how="inner")
    expected, missing = float(d["expected"].sum()), float(d["missing"].sum())
    density = d["trades_regular"].fillna(0) / d["expected"].clip(lower=1)
    bucket = pd.cut(density, [-1, 0.5, 1, 3, 10, float("inf")],
                    labels=["<0.5", "0.5-1", "1-3", "3-10", ">10"])
    by_density = {
        str(label): {"symbol_days": int(len(g)),
                     "missing_fraction": float(g["missing"].sum() / g["expected"].sum())}
        for label, g in d.groupby(bucket, observed=True) if g["expected"].sum() > 0
    }
    # Do the missing minutes carry volume? If the minute bars still add up to the daily volume,
    # the empty minutes had no trades; if not, trades were lost somewhere in the download.
    best = d[["vol_all", "vol_regular"]].sub(d["volume"], axis=0).abs().min(axis=1)
    d["vol_gap"] = best / d["volume"].where(d["volume"] > 0)
    gappy = d[(d["missing"] / d["expected"].clip(lower=1)) > 0.10]
    gappy_volume = {"days": int(len(gappy))}
    if len(gappy):
        gappy_volume.update({
            "median_rel_diff": float(gappy["vol_gap"].median()),
            "share_within_2pct": float((gappy["vol_gap"] <= 0.02).mean()),
        })
    # Where do the gaps live? By price (odd-lot trades do not build bars, and at high prices
    # nearly every trade is an odd lot) and by symbol.
    price_bins = pd.cut(d["price"], [0, 50, 100, 250, 1000, float("inf")],
                        labels=["<50", "50-100", "100-250", "250-1000", ">1000"])
    by_price = {
        str(label): {
            "symbol_days": int(len(g)),
            "missing_fraction": float(g["missing"].sum() / g["expected"].sum()),
            "median_volume_gap": float(g["vol_gap"].median()),
        }
        for label, g in d.groupby(price_bins, observed=True) if g["expected"].sum() > 0
    }
    per_symbol = d.groupby("symbol").agg(
        missing=("missing", "sum"), expected=("expected", "sum"), price=("price", "median"))
    per_symbol["fraction"] = per_symbol["missing"] / per_symbol["expected"]
    top_symbols = per_symbol.sort_values("fraction", ascending=False).head(10)
    symbol_quantiles = per_symbol["fraction"].quantile([0.5, 0.9, 0.99]).round(4).to_dict()
    worst = d[d["missing"] > 0].sort_values("missing", ascending=False).head(10)
    return {
        "symbol_days": len(d), "expected": int(expected), "missing": int(missing),
        "gappy_days_volume": gappy_volume,
        "by_price": by_price,
        "per_symbol_fraction_quantiles": {f"p{int(k * 100)}": v
                                          for k, v in symbol_quantiles.items()},
        "worst_symbols": [(sym, round(float(r.fraction), 4), round(float(r.price), 1))
                          for sym, r in top_symbols.iterrows()],
        "fraction": missing / expected if expected else None,
        "fraction_all": all_missing / all_expected if all_expected else None,
        "scope": "universe days" if members is not None else "all days",
        "by_trades_per_minute": by_density,
        "worst": [(r.symbol, str(r.day), int(r.missing), round(float(r.vol_gap), 3))
                  for r in worst.itertuples()],
    }


def price_jumps(
    daily: pd.DataFrame, threshold: float = 0.40, members: pd.DataFrame | None = None
) -> list[tuple[str, str, float]]:
    """Close-to-close moves beyond threshold in split-adjusted data: each needs a human look."""
    if daily.empty:
        return []
    d = daily.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    d["ret"] = d.groupby("symbol")["close"].pct_change()
    d["day"] = d["timestamp"].dt.tz_convert(NY).dt.date
    hit = d[d["ret"].abs() > threshold]
    if members is not None:  # only moves on days the stock was in the universe
        keep = members.loc[members["member"], ["symbol", "day"]]
        hit = hit.merge(keep, on=["symbol", "day"], how="inner")
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

    jumps = metrics["price_jumps"]  # jumps nobody has reviewed yet
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
