"""Rebuild split-adjusted prices as they were knowable at as_of. Pure: no I/O, no clock.

A split is applied to earlier bars only if its ex-date is on or before as_of, so a split that
happens later can never change a past decision.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from signalplat.utilities.clock import NY

PRICE_COLUMNS = ("open", "high", "low", "close", "vwap")


def adjust_for_splits(bars: pd.DataFrame, splits: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Return bars with prices divided, and volume multiplied, by the splits known at as_of."""
    out = bars.copy()
    if bars.empty or splits is None or splits.empty:
        return out
    factor = pd.Series(1.0, index=out.index)
    for s in splits.itertuples():
        # effective at midnight New York on the ex-date, when the new share count trades
        effective = pd.Timestamp(s.ex_date).tz_localize(NY)
        if effective > as_of:
            continue
        factor[(out["symbol"] == s.symbol) & (out["timestamp"] < effective)] *= float(s.ratio)
    for col in PRICE_COLUMNS:
        if col in out:
            out[col] = out[col] / factor
    if "volume" in out:
        out["volume"] = out["volume"] * factor
    return out
