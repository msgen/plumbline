import numpy as np
import pandas as pd
import pytest

from signalplat.engines.setups import add_targets, detect

CFG = {
    "window": {"start": "09:35", "end": "11:00"}, "rvol_min": 2.5,
    "rvol_applies_to": ["A_breakout", "C_catalyst_gap"],
    "stop": {"mode": "atr", "atr_mult": 1.0, "min_risk_pct": 0.002},
    "A_breakout": {"or_minutes": 15, "min_range_bars": 3, "require_above_vwap": True},
    "B_vwap_reclaim": {"band_sigma": 1.5, "lookback_bars": 30, "reclaim_bars": 10},
    "C_catalyst_gap": {"min_gap_pct": 0.03, "range_minutes": 5},
}
DAY = pd.Timestamp("2026-09-01").date()


def day_bars(closes, start=570, volume=1000.0, spread=0.1):
    n = len(closes)
    c = np.array(closes, float)
    return pd.DataFrame({
        "symbol": "AAA", "day": DAY, "mod": range(start, start + n), "open": c,
        "high": c + spread, "low": c - spread, "close": c, "volume": volume, "vwap": c})


def rvol_for(bars, sip, est):
    return pd.DataFrame({"symbol": "AAA", "day": DAY, "mod": bars["mod"],
                         "rvol_sip": sip, "rvol_est": est})


def ctx(atr_pct=0.02, gap_pct=0.0):
    return pd.DataFrame({"symbol": ["AAA"], "day": [DAY], "atr_pct": [atr_pct],
                         "gap_pct": [gap_pct]})


def breakout_day():
    # flat opening range (high 100.1) for 15 minutes, drift inside it, then a close above it
    closes = [100.0] * 15 + [100.05, 100.08, 100.1, 100.09, 100.5, 100.6, 100.7]
    return day_bars(closes)


def run(bars, rvol, c=None, **kw):
    return detect(bars, c if c is not None else ctx(), rvol, CFG, **kw)


def test_breakout_signal_bar_stop_and_variants():
    b = breakout_day()
    n = len(b)
    rv = rvol_for(b, [3.0] * n, [1.0] * n)           # SIP says busy, scaled IEX says quiet
    a = run(b, rv, strategies=("A_breakout",))
    got = {r.variant: r for r in a.itertuples()}
    assert set(got) == {"sip", "none"}               # iex_scaled stays below 2.5: no signal
    for r in got.values():
        assert r.signal_minute == 589                # first close above the 100.1 range high
        assert r.stop == pytest.approx(r.ref_price * (1 - 0.02))
    rv2 = rvol_for(b, [1.0] * n, [3.0] * n)
    v2 = set(run(b, rv2, strategies=("A_breakout",))["variant"])
    assert v2 == {"iex_scaled", "none"}


def test_rvol_nan_means_no_signal_for_the_volume_variants_only():
    b = breakout_day()
    rv = rvol_for(b, [np.nan] * len(b), [np.nan] * len(b))
    assert set(run(b, rv, strategies=("A_breakout",))["variant"]) == {"none"}


def test_nothing_fires_outside_the_window_or_before_the_range_closes():
    b = breakout_day()
    rv = rvol_for(b, [3.0] * len(b), [3.0] * len(b))
    cfg = {**CFG, "window": {"start": "09:35", "end": "09:48"}}
    assert detect(b, ctx(), rv, cfg, strategies=("A_breakout",)).empty
    early = day_bars([100.0] * 5 + [101.0] * 5)         # a "break" inside the first 15 minutes
    assert run(early, rvol_for(early, [3.0] * 10, [3.0] * 10), strategies=("A_breakout",)).empty


def test_vwap_reclaim():
    closes = [100.0] * 20 + [99.0, 98.5, 98.0, 97.8, 97.9, 98.5, 99.5, 101.0, 101.2]
    b = day_bars(closes, spread=0.05)
    b.loc[b["mod"] >= 590, "volume"] = 100.0           # keep VWAP near 100
    s = run(b, rvol_for(b, [np.nan] * len(b), [np.nan] * len(b)),
            strategies=("B_vwap_reclaim",))
    assert list(s["variant"]) == ["n/a"] and len(s) == 1
    row = s.iloc[0]
    assert row["signal_minute"] == 597 and row["ref_price"] == 101.0   # the close back above VWAP
    assert abs(row["stop"] - 101.0 * 0.98) < 1e-9                       # ATR stop, 2% below
    assert row["vwap_dev"] > 0


def test_gap_and_go_needs_a_gap():
    closes = [105.0, 105.2, 105.1, 105.3, 105.2, 105.6, 105.9, 106.0]
    b = day_bars(closes)
    rv = rvol_for(b, [3.0] * len(b), [3.0] * len(b))
    s = run(b, rv, ctx(gap_pct=0.05), strategies=("C_catalyst_gap",))
    assert set(s["variant"]) == {"sip", "iex_scaled", "none"}
    assert s["signal_minute"].iloc[0] == 575
    assert run(b, rv, ctx(gap_pct=0.01), strategies=("C_catalyst_gap",)).empty


def test_strategies_outside_rvol_applies_to_ignore_the_variants():
    closes = [100.0] * 25 + [99.0, 98.0, 97.5, 98.6, 100.5, 101.5]
    b = day_bars(closes, spread=0.05)
    rv = rvol_for(b, [np.nan] * len(b), [np.nan] * len(b))
    s = run(b, rv)
    assert set(s.loc[s["strategy"] == "B_vwap_reclaim", "variant"]) <= {"n/a"}


@pytest.mark.parametrize("seed", range(8))
def test_a_signal_never_depends_on_later_bars(seed):
    """Cut the day at every minute: signals found so far must match the full-day signals."""
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.0015, 120))
    b = day_bars(closes, spread=0.08)
    b["volume"] = rng.integers(100, 2000, len(b)).astype(float)
    rv = rvol_for(b, rng.uniform(0.5, 4.0, len(b)), rng.uniform(0.5, 4.0, len(b)))
    c = ctx(atr_pct=0.01, gap_pct=0.04)
    full = detect(b, c, rv, CFG)
    for cut in range(20, len(b), 7):
        part = detect(b.iloc[:cut], c, rv.iloc[:cut], CFG)
        last = b["mod"].iloc[cut - 1]
        want = full[full["signal_minute"] <= last].reset_index(drop=True)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), want, check_dtype=False)


def test_stops_stay_below_the_reference_and_targets_follow_r():
    b = breakout_day()
    rv = rvol_for(b, [3.0] * len(b), [3.0] * len(b))
    s = add_targets(run(b, rv, strategies=("A_breakout",)), [1.5, 2.0])
    assert (s["stop"] < s["ref_price"]).all()
    risk = s["ref_price"] - s["stop"]
    assert np.allclose(s["target"], s["ref_price"] + s["target_r"] * risk)
    assert run(b, rv, ctx(atr_pct=0.0001), strategies=("A_breakout",)).empty    # stop too tight
    assert run(b, rv, ctx(atr_pct=np.nan), strategies=("A_breakout",)).empty
