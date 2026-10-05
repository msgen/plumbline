"""Point-in-time liquidity and tradability filters. Pure: data and as_of come in as arguments."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd

EXCHANGES = ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")
# Alpaca's asset list has no ETF flag, so funds and odd instruments are excluded by name.
_EXCLUDED_NAME = re.compile(
    r"\b(?:ETF|ETN|FUND|TRUST|WARRANT|WARRANTS|RIGHT|RIGHTS|UNIT|UNITS|NOTES?|PREFERRED|"
    r"DEPOSITARY|ACQUISITION|PORTFOLIO|INDEX|SHARES)\b",
    re.IGNORECASE,
)
ATR_WINDOW = 14
VOLUME_WINDOW = 20


def eligible_assets(assets: pd.DataFrame, exchanges: tuple[str, ...] = EXCHANGES) -> list[str]:
    """Active, tradable common-stock-like symbols on major exchanges."""
    a = assets[(assets["status"] == "active") & assets["tradable"].astype(bool)]
    a = a[a["exchange"].isin(exchanges)]
    a = a[~a["name"].fillna("").str.contains(_EXCLUDED_NAME)]
    a = a[~a["symbol"].str.contains(r"[.\-/ ]")]  # class shares, preferreds and units
    return sorted(a["symbol"].unique())


def liquidity_table(daily: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Price, 20-day dollar volume and ATR% per symbol using bars available at as_of."""
    d = daily[daily["available_at"] <= as_of].sort_values(["symbol", "timestamp"])
    rows = []
    for sym, g in d.groupby("symbol"):
        if len(g) < ATR_WINDOW + 1:
            continue
        prev = g["close"].shift()
        tr = pd.concat([g["high"] - g["low"], (g["high"] - prev).abs(),
                        (g["low"] - prev).abs()], axis=1).max(axis=1)
        last = g.iloc[-1]
        rows.append({
            "symbol": sym, "close": float(last["close"]),
            "dollar_volume_20d": float((g["close"] * g["volume"]).tail(VOLUME_WINDOW).mean()),
            "atr_pct": float(tr.tail(ATR_WINDOW).mean() / last["close"]),
            "bars": len(g),
        })
    cols = ["symbol", "close", "dollar_volume_20d", "atr_pct", "bars"]
    return pd.DataFrame(rows, columns=cols)


def select_universe(
    daily: pd.DataFrame, as_of: datetime, cfg: dict[str, Any], n: int | None = None
) -> pd.DataFrame:
    """Symbols passing the price, dollar-volume and ATR filters, most liquid first.

    The spread filter needs quotes, which free history lacks; it is applied later from E2.
    """
    t = liquidity_table(daily, as_of)
    keep = t[(t["close"] > cfg["min_price"])
             & (t["dollar_volume_20d"] > cfg["min_dollar_volume_20d"])
             & (t["atr_pct"] > cfg["min_atr_pct"])]
    keep = keep.sort_values("dollar_volume_20d", ascending=False).reset_index(drop=True)
    return keep.head(n) if n else keep
