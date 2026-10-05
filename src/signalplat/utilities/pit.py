"""Point-in-time view: engines only ever see records with available_at <= as_of."""
from __future__ import annotations

from datetime import datetime

import pandas as pd


class FramePointInTimeView:
    def __init__(self, frames: dict[str, pd.DataFrame], as_of: datetime) -> None:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        self.as_of = as_of
        self._frames = {}
        for name, df in frames.items():
            if "available_at" not in df.columns:
                raise ValueError(f"frame {name!r} lacks an available_at column")
            self._frames[name] = df[df["available_at"] <= as_of].reset_index(drop=True)

    def get(self, name: str) -> pd.DataFrame:
        return self._frames[name].copy()


def poison_after(df: pd.DataFrame, as_of: datetime, value: float = 1e9) -> pd.DataFrame:
    """Replace numeric data after as_of with garbage. Used by leakage tests."""
    out = df.copy()
    late = out["available_at"] > as_of
    numeric = out.select_dtypes("number").columns
    out.loc[late, numeric] = value
    return out
