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
    level: float = 0.90, adjusted_level: float | None = None,
) -> dict[str, float]:
    """Mean R with a lower bound from resampling days with replacement.

    `level` is the one-sided confidence of the lower bound (0.90 -> the 10th percentile).
    `adjusted_level` adds a second, stricter bound from the same draws (see summarize_labels).
    """
    r = np.asarray(r, float)
    days = np.asarray(days)
    if len(r) == 0:
        return {"mean": float("nan"), "lower": float("nan"), "lower_adjusted": float("nan"),
                "n_days": 0}
    uniq, inverse = np.unique(days, return_inverse=True)
    sums = np.bincount(inverse, weights=r)
    counts = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    out = {
        "mean": float(r.mean()),
        "lower": float(np.quantile(means, 1 - level)),
        "n_days": int(len(uniq)),
    }
    if adjusted_level is not None:
        out["lower_adjusted"] = float(np.quantile(means, 1 - adjusted_level))
    return out


def summarize_labels(
    labels: pd.DataFrame, group_cols: list[str], gate: dict[str, float],
    n_boot: int = 10_000, seed: int = 0,
) -> pd.DataFrame:
    """One row per group: counts, outcome mix, mean R after costs and the E3 gate lines.

    Trades are labelled signals that were not `no_trade`. Expectancy includes time-outs.
    Every group is one trial. The gate uses a lower bound at a confidence raised for the number
    of groups (Bonferroni: 1 - (1 - level) / groups), so trying more combinations makes a pass
    harder. The unadjusted bound is shown for reference. Gross R (before spread and slippage)
    and the average stop distance are reported when the labels carry them.
    """
    groups = list(labels.groupby(group_cols, dropna=False, sort=True))
    level = gate["ci_level"]
    adjusted = 1 - (1 - level) / max(len(groups), 1)
    draws = int(min(max(n_boot, np.ceil(20 / (1 - adjusted))), 400_000))
    rows: list[dict[str, Any]] = []
    for key, g in groups:
        values = key if isinstance(key, tuple) else (key,)
        trades = g[g["outcome"] != "no_trade"]
        row = dict(zip(group_cols, values, strict=True))
        row.update({"signals": int(len(g)), "trades": int(len(trades)), "trials": len(groups)})
        if trades.empty:
            rows.append({**row, "gate": "no trades"})
            continue
        years = pd.to_datetime(trades["day"].astype(str)).dt.year
        by_year = trades.groupby(years)["r"].mean()
        boot = day_clustered_bootstrap(
            trades["r"].to_numpy(), trades["day"].to_numpy(), draws, seed, level, adjusted)
        mix = trades["outcome"].value_counts(normalize=True)
        row.update({
            "days": boot["n_days"], "target_rate": float(mix.get("target", 0.0)),
            "stop_rate": float(mix.get("stop", 0.0)),
            "timeout_rate": float(mix.get("timeout", 0.0)),
            "mean_r": boot["mean"], "mean_r_lower": boot["lower"],
            "mean_r_lower_adjusted": boot["lower_adjusted"],
            "years": int(len(by_year)), "share_years_positive": float((by_year > 0).mean()),
        })
        if "r_gross" in trades:
            row["mean_r_gross"] = float(trades["r_gross"].mean())
        if "risk_pct" in trades:
            row["mean_risk_pct"] = float(trades["risk_pct"].mean())
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
    if not row["mean_r_lower_adjusted"] > 0:
        problems.append(f"lower bound (adjusted for {row['trials']} trials) not above 0")
    if row["share_years_positive"] < gate["min_share_years_positive"]:
        problems.append(f"positive in {row['share_years_positive']:.0%} of years")
    return "pass" if not problems else "fail: " + "; ".join(problems)
