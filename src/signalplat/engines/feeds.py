"""Experiment E1 maths: can scaled IEX volume stand in for SIP volume near the open?

Pure functions over per-symbol-day opening summaries (v_HHMM and pv_HHMM columns, see
`ParquetBars.opening_volume`). No I/O, no clock.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

PRICE_BINS = [0, 50, 100, 250, 1000, float("inf")]
PRICE_LABELS = ["<50", "50-100", "100-250", "250-1000", ">1000"]


def label(minute: int) -> str:
    return f"{minute // 60:02d}{minute % 60:02d}"


def rvol_study(
    sip: pd.DataFrame, iex: pd.DataFrame, members: pd.DataFrame, minutes: list[int],
    trailing: int = 20, min_history: int = 10, flag: float = 2.5,
) -> dict[str, Any]:
    """Compare relative volume (RVOL) and VWAP from SIP with the IEX-based estimate.

    For each cut-off time T: RVOL = volume since 09:30 / the mean of the same figure over the
    previous `trailing` days. The IEX estimate scales today's IEX volume by that stock's
    trailing SIP/IEX volume ratio at T, using only earlier days, then divides by the same SIP
    baseline (history is available on the free plan). Judged on universe days only, but the
    trailing windows use every earlier day.
    """
    m = sip.merge(iex, on=["symbol", "day"], suffixes=("_sip", "_iex"))
    m = m.sort_values(["symbol", "day"]).reset_index(drop=True)
    keep = members.loc[members["member"], ["symbol", "day", "price"]]
    out: dict[str, Any] = {}
    for minute in minutes:
        lab = label(minute)
        s, i = m[f"v_{lab}_sip"], m[f"v_{lab}_iex"]
        g = m.assign(_s=s, _i=i).groupby("symbol", sort=False)

        def trailing_sum(col: str, g=g) -> pd.Series:
            return g[col].transform(
                lambda x: x.shift(1).rolling(trailing, min_periods=min_history).sum())

        sum_s, sum_i = trailing_sum("_s"), trailing_sum("_i")
        base = sum_s / g["_s"].transform(
            lambda x: x.shift(1).rolling(trailing, min_periods=min_history).count())
        ratio = sum_s / sum_i.where(sum_i > 0)
        rvol_sip = s / base.where(base > 0)
        rvol_est = i * ratio / base.where(base > 0)
        vw_sip = m[f"pv_{lab}_sip"] / s.where(s > 0)
        vw_iex = m[f"pv_{lab}_iex"] / i.where(i > 0)
        frame = pd.DataFrame({
            "symbol": m["symbol"], "day": m["day"], "rvol_sip": rvol_sip, "rvol_est": rvol_est,
            "rvol_err": (rvol_est - rvol_sip).abs() / rvol_sip.where(rvol_sip > 0),
            "vwap_bps": (vw_iex - vw_sip).abs() / vw_sip * 1e4,
        }).merge(keep, on=["symbol", "day"], how="inner")
        valid = frame.dropna(subset=["rvol_sip", "rvol_est"])
        out[lab] = _one_time(valid, frame, flag)
    return out


def _one_time(valid: pd.DataFrame, frame: pd.DataFrame, flag: float) -> dict[str, Any]:
    n = len(valid)
    if n == 0:
        return {"n": 0}
    f_sip, f_est = valid["rvol_sip"] >= flag, valid["rvol_est"] >= flag
    both, only_sip, only_est = (f_sip & f_est).sum(), (f_sip & ~f_est).sum(), (~f_sip & f_est).sum()
    flagged = both + only_sip + only_est
    bucket = pd.cut(valid["price"], PRICE_BINS, labels=PRICE_LABELS)
    return {
        "n": int(n),
        "median_rvol_error": float(valid["rvol_err"].median()),
        "p90_rvol_error": float(valid["rvol_err"].quantile(0.9)),
        "flag_agreement_all": float((f_sip == f_est).mean()),
        "flag_agreement_flagged": float(both / flagged) if flagged else None,
        "flag_precision": float(both / (both + only_est)) if both + only_est else None,
        "flag_recall": float(both / (both + only_sip)) if both + only_sip else None,
        "flagged_cases": int(flagged),
        "median_vwap_bps": float(valid["vwap_bps"].median()),
        "p90_vwap_bps": float(valid["vwap_bps"].quantile(0.9)),
        "median_rvol_error_by_price": {
            str(k): float(g["rvol_err"].median())
            for k, g in valid.groupby(bucket, observed=True)},
        "dropped_no_history": int(len(frame) - n),
    }


def evaluate_e1(
    study: dict[str, Any], gate: dict[str, float], error_label: str, flag_label: str
) -> dict[str, dict[str, Any]]:
    """Gate lines. Cases for flag agreement are the symbol-days either feed flags: otherwise
    the many days nobody flags would make any estimate look good."""

    def line(value, threshold, ok, op) -> dict[str, Any]:
        status = "n/a" if value is None else ("pass" if ok(value) else "fail")
        return {"value": value, "threshold": threshold, "op": op, "status": status}

    err = study.get(error_label, {}).get("median_rvol_error")
    agree = study.get(flag_label, {}).get("flag_agreement_flagged")
    return {
        f"median_rvol_error_{error_label}": line(
            err, gate["median_rvol_error"], lambda v: v < gate["median_rvol_error"], "<"),
        f"flag_agreement_{flag_label}": line(
            agree, gate["flag_agreement"], lambda v: v >= gate["flag_agreement"], ">="),
    }
