import numpy as np
import pandas as pd

from signalplat.engines.validation import day_clustered_bootstrap, summarize_labels

GATE = {"min_trades": 300, "min_days": 100, "min_share_years_positive": 0.6, "ci_level": 0.90}


def test_bootstrap_is_seeded_and_the_mean_is_exact():
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(200), 3)
    r = rng.normal(0.1, 1.0, 600)
    a = day_clustered_bootstrap(r, days, 2000, seed=5)
    assert a == day_clustered_bootstrap(r, days, 2000, seed=5)
    assert a["mean"] == r.mean() and a["n_days"] == 200
    assert a["lower"] < a["mean"]


def test_clustered_interval_is_wider_than_treating_trades_as_independent():
    rng = np.random.default_rng(2)
    n_days, per_day = 150, 8
    day_effect = rng.normal(0, 1.0, n_days)                   # all trades of a day move together
    r = (np.repeat(day_effect, per_day) + rng.normal(0, 0.3, n_days * per_day))
    days = np.repeat(np.arange(n_days), per_day)
    clustered = day_clustered_bootstrap(r, days, 4000, seed=3)
    independent = day_clustered_bootstrap(r, np.arange(len(r)), 4000, seed=3)
    assert r.mean() - clustered["lower"] > 1.5 * (r.mean() - independent["lower"])


def test_pure_noise_does_not_pass_the_gate():
    rng = np.random.default_rng(4)
    n = 4000
    day = pd.date_range("2024-01-01", periods=500, freq="D").date
    labels = pd.DataFrame({
        "strategy": "A", "variant": "none", "outcome": "timeout",
        "day": rng.choice(day, n), "r": rng.normal(0.0, 1.0, n)})
    out = summarize_labels(labels, ["strategy", "variant"], GATE, n_boot=2000)
    assert out.iloc[0]["gate"].startswith("fail")


def test_a_real_edge_passes_and_time_outs_count_in_the_mean():
    rng = np.random.default_rng(6)
    n = 3000
    day = pd.date_range("2023-01-01", periods=900, freq="D").date
    outcome = rng.choice(["target", "stop", "timeout"], n, p=[0.4, 0.4, 0.2])
    r = np.where(outcome == "target", 2.0, np.where(outcome == "stop", -1.0, 0.1))
    labels = pd.DataFrame({"strategy": "A", "variant": "sip", "outcome": outcome,
                           "day": rng.choice(day, n), "r": r})
    out = summarize_labels(labels, ["strategy", "variant"], GATE, n_boot=2000).iloc[0]
    assert out["gate"] == "pass", out["gate"]
    assert abs(out["mean_r"] - r.mean()) < 1e-12              # time-outs are in the average
    assert abs(out["timeout_rate"] - (outcome == "timeout").mean()) < 1e-12


def test_no_trade_rows_are_counted_as_signals_not_trades():
    labels = pd.DataFrame({
        "strategy": "A", "variant": "none", "day": [pd.Timestamp("2024-01-02").date()] * 3,
        "outcome": ["no_trade", "target", "stop"], "r": [np.nan, 2.0, -1.0]})
    out = summarize_labels(labels, ["strategy", "variant"], GATE, n_boot=200).iloc[0]
    assert out["signals"] == 3 and out["trades"] == 2
    assert out["gate"].startswith("fail: trades 2 < 300")
    empty = labels.assign(outcome="no_trade")
    assert summarize_labels(empty, ["strategy"], GATE, n_boot=200).iloc[0]["gate"] == "no trades"
