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
             & (t["close"] < cfg.get("max_price", float("inf")))
             & (t["dollar_volume_20d"] > cfg["min_dollar_volume_20d"])
             & (t["atr_pct"] > cfg["min_atr_pct"])]
    keep = keep.sort_values("dollar_volume_20d", ascending=False).reset_index(drop=True)
    return keep.head(n) if n else keep


def membership(daily: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """For every bar, whether its symbol was in the universe that day, from earlier bars only.

    Day t is judged on data through the previous bar's close. The price rule uses `raw_close`
    (the price level actually visible then); dollar volume and ATR% are unaffected by splits,
    so they use the adjusted columns. Returns symbol, day (New York date), member and price.
    """
    d = daily.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    g = d.groupby("symbol", sort=False)
    prev_close = g["close"].shift()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev_close).abs(),
                    (d["low"] - prev_close).abs()], axis=1).max(axis=1)
    d = d.assign(tr=tr, dollar_volume=d["close"] * d["volume"])
    g = d.groupby("symbol", sort=False)
    atr = g["tr"].transform(lambda s: s.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean())
    dv = g["dollar_volume"].transform(
        lambda s: s.rolling(VOLUME_WINDOW, min_periods=ATR_WINDOW + 1).mean())
    known = pd.DataFrame({
        "symbol": d["symbol"], "raw_close": d["raw_close"],
        "dv": dv, "atr_pct": atr / d["close"],
        "count": g.cumcount() + 1,
    })
    # what was known after the previous bar applies to this bar's day
    prev = known.groupby("symbol", sort=False)[["raw_close", "dv", "atr_pct", "count"]].shift()
    member = (
        (prev["count"] >= ATR_WINDOW + 1)
        & (prev["raw_close"] > cfg["min_price"])
        & (prev["raw_close"] < cfg.get("max_price", float("inf")))
        & (prev["dv"] > cfg["min_dollar_volume_20d"])
        & (prev["atr_pct"] > cfg["min_atr_pct"])
    ).fillna(False)
    day = d["timestamp"].dt.tz_convert("America/New_York").dt.date
    return pd.DataFrame({
        "symbol": d["symbol"], "day": day, "member": member,
        "price": prev["raw_close"],  # the price level visible before the day began
    })


def daily_context(daily: pd.DataFrame) -> pd.DataFrame:
    """Per symbol-day facts known when the day opens: ATR% from earlier bars, and the gap.

    `daily` holds split-adjusted bars (symbol, timestamp, open, high, low, close). ATR% uses
    bars through the previous day only; the gap is today's open over the previous close, both
    on the same adjusted scale. Returns symbol, day, atr_pct, gap_pct, prev_close.
    """
    d = daily.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    g = d.groupby("symbol", sort=False)
    prev_close = g["close"].shift()
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev_close).abs(),
                    (d["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.groupby(d["symbol"], sort=False).transform(
        lambda s: s.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean())
    atr_pct_close = atr / d["close"]
    day = d["timestamp"].dt.tz_convert("America/New_York").dt.date
    return pd.DataFrame({
        "symbol": d["symbol"], "day": day,
        "atr_pct": atr_pct_close.groupby(d["symbol"], sort=False).shift(),
        "gap_pct": d["open"] / prev_close - 1,
        "prev_close": prev_close,
    })
