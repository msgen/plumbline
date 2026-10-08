"""Triple-barrier labels with entry delay and costs. Pure: bars and signals come in as frames.

Each signal becomes one of four outcomes:
  target   the target was reached first
  stop     the stop was reached first (a bar that touches both counts as the stop)
  timeout  neither barrier was reached inside the time limit; the trade exits at the close
  no_trade the price had already run past the stop or the target when the delayed entry filled

R is profit divided by the risk actually taken at the fill (fill minus stop), after costs.
Time-outs keep their realised R: expectancy must include them (plan amendment 2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

SESSION_END_MINUTE = 16 * 60


@dataclass(frozen=True)
class LabelParams:
    delay_seconds: int = 60          # from the signal bar's close to the fill
    time_limit_minutes: int = 60
    half_spread: float = 0.0005      # fraction of price paid to cross the spread
    slippage: float = 0.0002         # extra adverse fill, as a fraction of price
    session_end_minute: int = SESSION_END_MINUTE


OUTPUT_COLUMNS = [
    "outcome", "entry_minute", "entry_gap_minutes", "entry_price", "exit_minute",
    "exit_price", "r", "minutes_held",
]


def label_signals(signals: pd.DataFrame, bars: pd.DataFrame, params: LabelParams) -> pd.DataFrame:
    """Label each signal using only bars from the entry minute onward.

    signals: symbol, day, signal_minute (minute of the bar whose close triggered it),
             stop, target (price levels). Extra columns are carried through.
    bars:    symbol, day, mod (minute of day), open, high, low, close, sorted by mod.
    """
    if signals.empty:
        return signals.assign(**{c: pd.Series(dtype="float64") for c in OUTPUT_COLUMNS})
    arrays = {
        key: (g["mod"].to_numpy(), g["open"].to_numpy(), g["high"].to_numpy(),
              g["low"].to_numpy(), g["close"].to_numpy())
        for key, g in bars.groupby(["symbol", "day"], sort=False)
    }
    rows = []
    for sig in signals.itertuples(index=False):
        a = arrays.get((sig.symbol, sig.day))
        rows.append(_label_one(sig, a, params) if a is not None else _no_trade("no_bars"))
    out = pd.DataFrame(rows, index=signals.index)
    return pd.concat([signals, out], axis=1)


def _no_trade(reason: str) -> dict:
    return {"outcome": "no_trade", "entry_minute": np.nan, "entry_gap_minutes": np.nan,
            "entry_price": np.nan, "exit_minute": np.nan, "exit_price": np.nan,
            "r": np.nan, "minutes_held": np.nan, "skip_reason": reason}


def _label_one(sig, arrays, p: LabelParams) -> dict:
    mods, opens, highs, lows, closes = arrays
    wanted = sig.signal_minute + 1 + math.ceil(p.delay_seconds / 60)
    i = int(np.searchsorted(mods, wanted))
    if i >= len(mods) or mods[i] >= p.session_end_minute:
        return _no_trade("no_bar_after_delay")
    cost = p.half_spread + p.slippage
    fill = opens[i] * (1 + cost)
    if fill <= sig.stop:
        return _no_trade("through_stop")
    if fill >= sig.target:
        return _no_trade("through_target")
    risk = fill - sig.stop
    horizon = min(mods[i] + p.time_limit_minutes, p.session_end_minute)
    last = i
    for j in range(i, len(mods)):
        if mods[j] >= horizon:
            break
        last = j
        gapped = j > i and opens[j] <= sig.stop
        if gapped or lows[j] <= sig.stop:
            exit_price = (opens[j] if gapped else sig.stop) * (1 - p.slippage)
            return _done("stop", mods, i, j, wanted, fill, exit_price, risk)
        if highs[j] >= sig.target * (1 + p.half_spread):
            return _done("target", mods, i, j, wanted, fill, sig.target, risk)
    exit_price = closes[last] * (1 - cost)
    return _done("timeout", mods, i, last, wanted, fill, exit_price, risk)


def _done(outcome, mods, i, j, wanted, fill, exit_price, risk) -> dict:
    return {
        "outcome": outcome, "entry_minute": float(mods[i]),
        "entry_gap_minutes": float(mods[i] - wanted), "entry_price": float(fill),
        "exit_minute": float(mods[j]), "exit_price": float(exit_price),
        "r": float((exit_price - fill) / risk), "minutes_held": float(mods[j] + 1 - mods[i]),
        "skip_reason": None,
    }
