"""Setup detectors A, B and C, each with three volume-trigger variants. Pure.

A bar's signal uses only that bar and earlier ones. Prices come from the research bars; the
volume condition (relative volume, RVOL) comes in three variants, matching the E1 decision:
  sip         true consolidated RVOL (what a paid real-time feed would give)
  iex_scaled  RVOL estimated from free IEX volume (what free live data can give)
  none        no volume condition at all
Strategies not listed in `rvol_applies_to` ignore the variant and carry variant "n/a".
At most one signal per symbol, day, strategy and variant (the first).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STRATEGIES = ("A_breakout", "B_vwap_reclaim", "C_catalyst_gap")
VARIANTS = ("sip", "iex_scaled", "none")
OPEN = 570
SIGNAL_COLUMNS = [
    "symbol", "day", "strategy", "variant", "signal_minute", "ref_price", "stop", "atr_pct",
    "gap_pct", "rvol_sip", "rvol_est", "vwap_dev",
]


def _minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


def detect(
    bars: pd.DataFrame, ctx: pd.DataFrame, rvol: pd.DataFrame, cfg: dict[str, Any],
    strategies: tuple[str, ...] = STRATEGIES, variants: tuple[str, ...] = VARIANTS,
) -> pd.DataFrame:
    """All signals for the symbol-days in `ctx`.

    bars: symbol, day, mod, open, high, low, close, volume, vwap (regular session, sorted).
    ctx:  symbol, day, atr_pct, gap_pct for the days to scan.
    rvol: symbol, day, mod, rvol_sip, rvol_est (from `features.rvol_grids`).
    """
    wanted = ctx.set_index(["symbol", "day"])
    merged = bars.merge(rvol, on=["symbol", "day", "mod"], how="left")
    rows: list[dict] = []
    applies = set(cfg.get("rvol_applies_to", []))
    for key, g in merged.groupby(["symbol", "day"], sort=False):
        if key not in wanted.index:
            continue
        c = wanted.loc[key]
        day = _Day(g)
        for strategy in strategies:
            for variant in (variants if strategy in applies else ("n/a",)):
                row = _detect_one(strategy, variant, day, c, cfg)
                if row is not None:
                    rows.append({"symbol": key[0], "day": key[1], **row})
    if not rows:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    return pd.DataFrame(rows)[SIGNAL_COLUMNS]


class _Day:
    """One symbol-day of bars as numpy arrays, plus the running VWAP."""

    def __init__(self, g: pd.DataFrame) -> None:
        self.m = g["mod"].to_numpy()
        self.o, self.h = g["open"].to_numpy(float), g["high"].to_numpy(float)
        self.l, self.c = g["low"].to_numpy(float), g["close"].to_numpy(float)
        vol = g["volume"].to_numpy(float)
        px = np.where(np.isfinite(g["vwap"].to_numpy(float)), g["vwap"].to_numpy(float), self.c)
        cv, cpv = np.cumsum(vol), np.cumsum(px * vol)
        self.vwap = np.where(cv > 0, cpv / np.where(cv > 0, cv, 1.0), np.nan)
        self.rvol_sip = g["rvol_sip"].to_numpy(float)
        self.rvol_est = g["rvol_est"].to_numpy(float)


def _detect_one(strategy, variant, d: _Day, ctx, cfg) -> dict | None:
    win_lo, win_hi = _minutes(cfg["window"]["start"]), _minutes(cfg["window"]["end"])
    ok = (d.m >= win_lo) & (d.m < win_hi)
    if variant == "sip":
        ok &= d.rvol_sip >= cfg["rvol_min"]            # NaN compares False: no history, no signal
    elif variant == "iex_scaled":
        ok &= d.rvol_est >= cfg["rvol_min"]
    atr_pct = float(ctx["atr_pct"])
    if strategy == "A_breakout":
        found = _breakout(d, ok, cfg["A_breakout"])
    elif strategy == "B_vwap_reclaim":
        found = _vwap_reclaim(d, ok, cfg["B_vwap_reclaim"])
    else:
        found = _gap_and_go(d, ok, float(ctx["gap_pct"]), cfg["C_catalyst_gap"])
    if found is None:
        return None
    i, structure_stop = found
    ref = float(d.c[i])
    stop = _stop(ref, atr_pct, structure_stop, cfg["stop"])
    if stop is None:
        return None
    return {
        "strategy": strategy, "variant": variant, "signal_minute": int(d.m[i]),
        "ref_price": ref, "stop": stop, "atr_pct": atr_pct, "gap_pct": float(ctx["gap_pct"]),
        "rvol_sip": float(d.rvol_sip[i]), "rvol_est": float(d.rvol_est[i]),
        "vwap_dev": float(ref / d.vwap[i] - 1) if np.isfinite(d.vwap[i]) else np.nan,
    }


def _stop(ref: float, atr_pct: float, structure: float | None, cfg: dict) -> float | None:
    atr_stop = ref * (1 - cfg["atr_mult"] * atr_pct) if np.isfinite(atr_pct) else None
    stop = structure if cfg["mode"] == "structure" and structure is not None else atr_stop
    if stop is None or not np.isfinite(stop) or stop >= ref:
        stop = atr_stop
    if stop is None or stop >= ref or (ref - stop) / ref < cfg.get("min_risk_pct", 0.002):
        return None
    return float(stop)


def _first(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def _breakout(d: _Day, ok: np.ndarray, cfg: dict):
    """Close above the opening-range high, after the range has closed, and above VWAP."""
    end = OPEN + cfg["or_minutes"]
    rng = (d.m >= OPEN) & (d.m < end)
    if rng.sum() < cfg.get("min_range_bars", 3):
        return None
    cand = ok & (d.m >= end) & (d.c > d.h[rng].max())
    if cfg.get("require_above_vwap", True):
        cand &= d.c > d.vwap
    i = _first(cand)
    return None if i is None else (i, float(d.l[rng].min()))


def _vwap_reclaim(d: _Day, ok: np.ndarray, cfg: dict):
    """Close back above VWAP after trading well below it (in units of its own volatility)."""
    dev = pd.Series(d.c - d.vwap)
    sigma = dev.rolling(cfg["lookback_bars"], min_periods=cfg["lookback_bars"] // 2).std()
    dipped = dev.shift(1).rolling(cfg["reclaim_bars"], min_periods=1).min()
    cross = (dev > 0) & (dev.shift(1) <= 0)
    cand = ok & (cross & (dipped < -cfg["band_sigma"] * sigma.shift(1))).to_numpy()
    i = _first(cand)
    if i is None:
        return None
    lo = float(np.min(d.l[max(0, i - cfg["reclaim_bars"]): i + 1]))
    return i, lo


def _gap_and_go(d: _Day, ok: np.ndarray, gap_pct: float, cfg: dict):
    """Gap up at the open, then a close above the high of the first few minutes.

    The catalyst check (news, filings) joins in experiments E9 and E10; this is the price part.
    """
    if not np.isfinite(gap_pct) or gap_pct < cfg["min_gap_pct"]:
        return None
    end = OPEN + cfg["range_minutes"]
    rng = (d.m >= OPEN) & (d.m < end)
    if not rng.any():
        return None
    i = _first(ok & (d.m >= end) & (d.c > d.h[rng].max()))
    return None if i is None else (i, float(d.l[rng].min()))


def add_targets(signals: pd.DataFrame, target_rs: list[float]) -> pd.DataFrame:
    """One row per target multiple: target = reference + R x (reference - stop)."""
    parts = []
    for r in target_rs:
        part = signals.copy()
        part["target_r"] = r
        part["target"] = part["ref_price"] + r * (part["ref_price"] - part["stop"])
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else signals
