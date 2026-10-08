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


def test_more_trials_make_the_same_result_harder_to_pass():
    rng = np.random.default_rng(8)
    n = 1500
    day = pd.date_range("2023-01-01", periods=600, freq="D").date
    r = rng.normal(0.08, 1.0, n)                               # a weak edge
    one = pd.DataFrame({"g": "a", "outcome": "timeout", "day": rng.choice(day, n), "r": r})
    many = pd.concat([one.assign(g=f"g{i}") for i in range(40)], ignore_index=True)
    a = summarize_labels(one, ["g"], GATE, n_boot=4000, seed=1).iloc[0]
    b = summarize_labels(many, ["g"], GATE, n_boot=4000, seed=1).iloc[0]
    assert a["trials"] == 1 and b["trials"] == 40
    assert abs(a["mean_r_lower"] - a["mean_r_lower_adjusted"]) < 1e-12     # nothing to adjust
    assert b["mean_r_lower_adjusted"] < b["mean_r_lower"]                  # stricter bound


def test_gross_r_and_risk_are_summarised_when_present():
    labels = pd.DataFrame({
        "g": "a", "day": [pd.Timestamp("2024-01-02").date()] * 4,
        "outcome": ["target", "stop", "timeout", "timeout"],
        "r": [1.9, -1.1, 0.0, -0.1], "r_gross": [2.0, -1.0, 0.1, 0.0],
        "risk_pct": [0.02, 0.02, 0.03, 0.03]})
    out = summarize_labels(labels, ["g"], GATE, n_boot=100).iloc[0]
    assert abs(out["mean_r_gross"] - 0.275) < 1e-12 and abs(out["mean_risk_pct"] - 0.025) < 1e-12
