"""Statistics for judging labelled signals. Pure functions, seeded randomness.

Signals on the same day (or around the same event) are not independent, so the confidence
interval resamples whole trading days (plan amendment 4).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def day_clustered_bootstrap(
    r: np.ndarray, days: np.ndarray, n_boot: int = 10_000, seed: int = 0,
    level: float = 0.90,
) -> dict[str, float]:
    """Mean R with a lower bound from resampling days with replacement.

    `level` is the one-sided confidence of the lower bound (0.90 -> the 10th percentile).
    """
    r = np.asarray(r, float)
    days = np.asarray(days)
    if len(r) == 0:
        return {"mean": float("nan"), "lower": float("nan"), "n_days": 0}
    uniq, inverse = np.unique(days, return_inverse=True)
    sums = np.bincount(inverse, weights=r)
    counts = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return {
        "mean": float(r.mean()),
        "lower": float(np.quantile(means, 1 - level)),
        "n_days": int(len(uniq)),
    }


def summarize_labels(
    labels: pd.DataFrame, group_cols: list[str], gate: dict[str, float],
    n_boot: int = 10_000, seed: int = 0,
) -> pd.DataFrame:
    """One row per group: counts, outcome mix, mean R after costs and the E3 gate lines.

    Trades are labelled signals that were not `no_trade`. Expectancy includes time-outs.
    """
    rows: list[dict[str, Any]] = []
    for key, g in labels.groupby(group_cols, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        trades = g[g["outcome"] != "no_trade"]
        row = dict(zip(group_cols, values, strict=True))
        row.update({"signals": int(len(g)), "trades": int(len(trades))})
        if trades.empty:
            rows.append({**row, "gate": "no trades"})
            continue
        years = pd.to_datetime(trades["day"].astype(str)).dt.year
        by_year = trades.groupby(years)["r"].mean()
        boot = day_clustered_bootstrap(trades["r"].to_numpy(), trades["day"].to_numpy(),
                                       n_boot, seed, gate["ci_level"])
        mix = trades["outcome"].value_counts(normalize=True)
        row.update({
            "days": boot["n_days"], "target_rate": float(mix.get("target", 0.0)),
            "stop_rate": float(mix.get("stop", 0.0)),
            "timeout_rate": float(mix.get("timeout", 0.0)),
            "mean_r": boot["mean"], "mean_r_lower": boot["lower"],
            "years": int(len(by_year)), "share_years_positive": float((by_year > 0).mean()),
        })
        row["gate"] = _gate(row, gate)
        rows.append(row)
    return pd.DataFrame(rows)


def _gate(row: dict[str, Any], gate: dict[str, float]) -> str:
    """E3 gate: enough signals on enough days, a positive clustered lower bound, most years."""
    problems = []
    if row["trades"] < gate["min_trades"]:
        problems.append(f"trades {row['trades']} < {gate['min_trades']}")
    if row["days"] < gate["min_days"]:
        problems.append(f"days {row['days']} < {gate['min_days']}")
    if not row["mean_r_lower"] > 0:
        problems.append("lower bound not above 0")
    if row["share_years_positive"] < gate["min_share_years_positive"]:
        problems.append(f"positive in {row['share_years_positive']:.0%} of years")
    return "pass" if not problems else "fail: " + "; ".join(problems)
