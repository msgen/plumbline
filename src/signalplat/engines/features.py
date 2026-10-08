"""Intraday relative volume (RVOL) by minute, from SIP and from scaled IEX.

Pure functions. The two estimates mirror experiment E1:
  rvol_sip  = volume so far / the same figure averaged over the previous days (true feed)
  rvol_est  = IEX volume so far, scaled by the stock's trailing SIP/IEX ratio, over the same
              SIP baseline: what free live data could deliver
Only earlier days enter a baseline or a ratio, never the day itself.
"""
from __future__ import annotations

import pandas as pd

OPEN, CLOSE = 570, 960


def cumulative_volume(minutes: pd.DataFrame, until: int) -> dict[str, pd.DataFrame]:
    """Per symbol: days x minute-of-day grid of volume since 09:30 (absent minutes count 0)."""
    cols = list(range(OPEN, until))
    out: dict[str, pd.DataFrame] = {}
    sel = minutes[(minutes["mod"] >= OPEN) & (minutes["mod"] < until)]
    for sym, g in sel.groupby("symbol", sort=False):
        grid = g.pivot_table(index="day", columns="mod", values="volume", aggfunc="sum")
        out[sym] = grid.reindex(columns=cols).fillna(0.0).sort_index().cumsum(axis=1)
    return out


def rvol_grids(
    sip: pd.DataFrame, iex: pd.DataFrame, until: int = 720, trailing: int = 20,
    min_history: int = 10,
) -> pd.DataFrame:
    """RVOL at the close of every minute from 09:30 to `until`, per symbol and day.

    sip / iex: symbol, day, mod, volume (regular-session minute bars).
    Returns symbol, day, mod, rvol_sip, rvol_est; `mod` is the bar's minute, so the value is
    known at the end of that bar.
    """
    cs, ci = cumulative_volume(sip, until), cumulative_volume(iex, until)
    frames = []
    for sym, s in cs.items():
        i = ci.get(sym)
        i = (i.reindex(s.index).fillna(0.0) if i is not None
             else pd.DataFrame(0.0, index=s.index, columns=s.columns))
        past_s = s.shift(1).rolling(trailing, min_periods=min_history)
        past_i = i.shift(1).rolling(trailing, min_periods=min_history)
        base = past_s.mean()
        sum_s, sum_i = past_s.sum(), past_i.sum()
        ratio = sum_s / sum_i.where(sum_i > 0)
        base = base.where(base > 0)
        rvol_sip = s / base
        rvol_est = i * ratio / base
        f = pd.DataFrame({
            "rvol_sip": rvol_sip.stack(future_stack=True),
            "rvol_est": rvol_est.stack(future_stack=True),
        }).reset_index()
        f.columns = ["day", "mod", "rvol_sip", "rvol_est"]
        f.insert(0, "symbol", sym)
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["symbol", "day", "mod", "rvol_sip", "rvol_est"])
    return pd.concat(frames, ignore_index=True)
