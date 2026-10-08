import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from signalplat.engines.labeling import LabelParams, label_signals

FREE = LabelParams(delay_seconds=0, time_limit_minutes=60, half_spread=0.0, slippage=0.0)


def bars(rows, start=585):
    """rows: (open, high, low, close) per consecutive minute from `start`."""
    return pd.DataFrame({
        "symbol": "AAA", "day": "d", "mod": range(start, start + len(rows)),
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows]})


def signal(minute=584, stop=99.0, target=102.0):
    return pd.DataFrame({"symbol": ["AAA"], "day": ["d"], "signal_minute": [minute],
                         "stop": [stop], "target": [target]})


def one(rows, sig=None, params=FREE, **kw):
    return label_signals(sig if sig is not None else signal(**kw), bars(rows), params).iloc[0]


def test_target_hit_is_plus_2r():
    r = one([(100, 100.5, 99.5, 100), (100, 102.5, 100, 102)])
    assert r["outcome"] == "target" and r["r"] == 2.0
    assert r["entry_minute"] == 585 and r["exit_minute"] == 586


def test_stop_hit_is_minus_1r():
    r = one([(100, 100.5, 99.5, 100), (100, 100.2, 98.5, 99)])
    assert r["outcome"] == "stop" and r["r"] == -1.0


def test_a_bar_touching_both_barriers_counts_as_the_stop():
    r = one([(100, 103, 98, 100)])
    assert r["outcome"] == "stop"


def test_gap_down_through_the_stop_fills_at_the_open():
    r = one([(100, 100.5, 99.5, 100), (97, 98, 96, 97)])
    assert r["outcome"] == "stop" and r["exit_price"] == 97.0 and r["r"] == -3.0


def test_timeout_exits_at_the_close_and_keeps_its_r():
    rows = [(100, 100.5, 99.5, 100)] * 59 + [(100, 100.6, 99.6, 100.5)] + [(100, 105, 99.5, 104)]
    r = one(rows)                          # the 61st bar (which would reach the target) is late
    assert r["outcome"] == "timeout" and r["r"] == 0.5 and r["minutes_held"] == 60


def test_delay_moves_the_entry_to_a_later_bar():
    rows = [(100, 100.5, 99.5, 100), (101, 101.5, 100.5, 101), (102, 102.5, 101.5, 102)]
    now = one(rows, target=105.0)
    late = one(rows, target=105.0, params=LabelParams(61, 60, 0, 0))   # two bars after the signal
    # signal bar closes at 585:00; +61 s is 586:01, so the next bar that has started is 587
    assert now["entry_price"] == 100.0 and late["entry_price"] == 102.0
    assert now["entry_minute"] == 585 and late["entry_minute"] == 587
    half = one(rows, target=105.0, params=LabelParams(30, 60, 0, 0))  # 585:30 waits for 586
    assert half["entry_minute"] == 586 and half["entry_price"] == 101.0
    assert one(rows, target=105.0, params=LabelParams(0, 60, 0, 0))["entry_minute"] == 585


def test_costs_widen_the_fill_and_cut_the_exit():
    p = LabelParams(0, 60, half_spread=0.001, slippage=0.0005)
    r = one([(100, 100.5, 99.5, 100)] * 2, params=p)
    assert abs(r["entry_price"] - 100 * 1.0015) < 1e-9
    assert abs(r["exit_price"] - 100 * (1 - 0.0015)) < 1e-9
    assert r["r"] < 0


def test_target_needs_price_to_trade_through_the_spread():
    p = LabelParams(0, 60, half_spread=0.01, slippage=0.0)
    r = one([(100, 102.5, 100, 102)] * 3, params=p)     # high 102.5 < 102 * 1.01
    assert r["outcome"] == "timeout"
    r = one([(100, 103.1, 100, 103)], params=p)         # 103.1 >= 102 * 1.01
    assert r["outcome"] == "target"


def test_no_trade_when_the_entry_is_already_past_a_barrier():
    assert one([(98.5, 99, 98, 98.6)])["outcome"] == "no_trade"             # through the stop
    assert one([(103, 104, 102, 103)])["outcome"] == "no_trade"             # through the target
    late = label_signals(signal(minute=900), bars([(100, 101, 99.5, 100)]), FREE).iloc[0]
    assert late["outcome"] == "no_trade"
    missing = label_signals(signal(), bars([(100, 101, 99.5, 100)]).assign(symbol="ZZZ"), FREE)
    assert missing.iloc[0]["skip_reason"] == "no_bars"


def test_a_missing_minute_uses_the_next_bar_and_records_the_gap():
    rows = bars([(100, 100.5, 99.5, 100)] * 3)
    rows = rows[rows["mod"] != 585]                  # no trades in the entry minute
    r = label_signals(signal(), rows, FREE).iloc[0]
    assert r["entry_minute"] == 586 and r["entry_gap_minutes"] == 1


def test_session_close_ends_the_trade():
    rows = bars([(100, 100.4, 99.6, 100)] * 5, start=957)      # 15:57 to 16:01
    r = label_signals(signal(minute=956), rows, FREE).iloc[0]
    assert r["outcome"] == "timeout" and r["exit_minute"] == 959


def test_bars_before_the_entry_cannot_change_the_label():
    rows = [(100, 100.5, 99.5, 100), (100, 102.5, 100, 102)]
    clean = bars(rows, start=585)
    before = bars([(50, 500, 1, 60)] * 5, start=580)       # garbage before the signal
    base = label_signals(signal(), clean, FREE)
    poisoned = label_signals(signal(), pd.concat([before, clean]), FREE)
    pd.testing.assert_frame_equal(base, poisoned)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.floats(-0.02, 0.02), st.floats(0, 0.01), st.floats(0, 0.01)),
                min_size=3, max_size=40),
       st.integers(0, 120), st.floats(0.5, 3.0))
def test_random_walks_obey_the_barrier_rules(steps, delay, target_r):
    price, rows = 100.0, []
    for move, up, down in steps:
        o = price
        c = o * (1 + move)
        rows.append((o, max(o, c) * (1 + up), min(o, c) * (1 - down), c))
        price = c
    stop = 99.0
    sig = signal(stop=stop, target=100.0 + target_r)
    p = LabelParams(delay, 30, 0.0003, 0.0001)
    out = label_signals(sig, bars(rows), p).iloc[0]
    if out["outcome"] == "no_trade":
        assert np.isnan(out["r"])
        return
    assert out["entry_price"] > stop and out["entry_price"] < sig["target"].iloc[0]
    assert out["exit_minute"] >= out["entry_minute"]
    assert out["minutes_held"] <= 30 + 1
    if out["outcome"] == "target":
        assert out["r"] > 0
    if out["outcome"] == "stop":
        assert out["r"] <= 0
