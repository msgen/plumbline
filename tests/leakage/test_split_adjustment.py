"""A split after as_of must not change what the universe saw at as_of."""
from datetime import UTC, datetime

import pandas as pd

from signalplat.engines.adjust import adjust_for_splits
from signalplat.engines.universe import select_universe

CFG = {"min_price": 10.0, "min_dollar_volume_20d": 1_000_000, "min_atr_pct": 0.0}
NY = "America/New_York"


def raw_daily(symbol, closes, start):
    ts = pd.bdate_range(start, periods=len(closes), tz=NY).tz_convert("UTC")
    return pd.DataFrame({
        "symbol": symbol, "timestamp": ts, "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes, "vwap": closes, "volume": 1_000_000.0,
        "available_at": ts + pd.Timedelta("1D")})


def splits(ex_date, ratio):
    return pd.DataFrame({"symbol": ["AAA"], "ex_date": [ex_date], "ratio": [ratio]})


def test_adjustment_math_forward_and_reverse():
    d = raw_daily("AAA", [100.0, 100.0, 25.0, 25.0], "2026-03-02")  # 4-for-1 on 03-04
    out = adjust_for_splits(d, splits("2026-03-04", 4.0), datetime(2026, 3, 10, tzinfo=UTC))
    assert list(out["close"]) == [25.0, 25.0, 25.0, 25.0]
    assert list(out["volume"]) == [4e6, 4e6, 1e6, 1e6]
    assert list(d["close"]) == [100.0, 100.0, 25.0, 25.0]  # input is not modified
    r = raw_daily("AAA", [2.0, 2.0, 20.0], "2026-03-02")  # 1-for-10 reverse on 03-04
    out = adjust_for_splits(r, splits("2026-03-04", 0.1), datetime(2026, 3, 10, tzinfo=UTC))
    assert [round(x, 9) for x in out["close"]] == [20.0, 20.0, 20.0]


def test_future_reverse_split_cannot_change_the_price_filter_at_as_of():
    # $8 stock (fails the $10 filter) that later reverse-splits 1-for-10 and trades at $80
    d = raw_daily("AAA", [8.0] * 30 + [80.0] * 10, "2026-01-05")
    ex = "2026-02-17"  # first day of $80 prices
    s = splits(ex, 0.1)
    as_of = datetime(2026, 2, 16, 12, 0, tzinfo=UTC)  # before the split
    before = select_universe(adjust_for_splits(d, s, as_of), as_of, CFG)
    assert before.empty  # at the time it really was an $8 stock
    # the same decision with the split poisoned or removed must be identical
    poisoned = splits(ex, 1e6)
    same = select_universe(adjust_for_splits(d, poisoned, as_of), as_of, CFG)
    pd.testing.assert_frame_equal(before, same)
    # after the ex-date the split is known and the old prices are restated
    later = datetime(2026, 3, 1, tzinfo=UTC)
    after = select_universe(adjust_for_splits(d, s, later), later, CFG)
    assert list(after["symbol"]) == ["AAA"] and after.iloc[0]["close"] == 80.0
