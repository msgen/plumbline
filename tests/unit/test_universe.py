from datetime import UTC, datetime

import pandas as pd

from signalplat.engines.universe import eligible_assets, liquidity_table, select_universe

CFG = {"min_price": 10.0, "min_dollar_volume_20d": 1_000_000, "min_atr_pct": 0.02}


def bars(symbol, closes, highs=None, lows=None, volume=100_000, start="2026-08-03"):
    n = len(closes)
    ts = pd.bdate_range(start, periods=n, tz="UTC")
    return pd.DataFrame({
        "symbol": symbol, "timestamp": ts, "close": closes,
        "high": highs if highs is not None else [c + 1 for c in closes],
        "low": lows if lows is not None else [c - 1 for c in closes],
        "volume": volume, "available_at": ts + pd.Timedelta("1D")})


def test_atr_pct_and_dollar_volume_match_hand_calculation():
    # flat price 100, each bar high 102 low 98 and no gaps: true range is 4 every day
    d = bars("AAA", [100.0] * 25, highs=[102.0] * 25, lows=[98.0] * 25, volume=50_000)
    t = liquidity_table(d, datetime(2027, 1, 1, tzinfo=UTC)).iloc[0]
    assert abs(t["atr_pct"] - 0.04) < 1e-12
    assert t["dollar_volume_20d"] == 100.0 * 50_000


def test_gap_widens_true_range():
    closes = [100.0] * 20 + [110.0]  # last bar gaps up; its range is only 2 but |h - prev| = 11
    highs = [101.0] * 20 + [111.0]
    lows = [99.0] * 20 + [109.0]
    t = liquidity_table(bars("AAA", closes, highs, lows), datetime(2027, 1, 1, tzinfo=UTC))
    expected = (13 * 2 + 11) / 14 / 110.0
    assert abs(t.iloc[0]["atr_pct"] - expected) < 1e-12


def test_universe_is_point_in_time_and_poison_proof():
    d = bars("AAA", [50.0] * 30)
    as_of = d["available_at"].iloc[24]
    clean = liquidity_table(d, as_of)
    poisoned = d.copy()
    late = poisoned["available_at"] > as_of
    poisoned.loc[late, ["close", "high", "low", "volume"]] = 1e9
    pd.testing.assert_frame_equal(clean, liquidity_table(poisoned, as_of))
    assert clean.iloc[0]["bars"] == 25


def test_filters_and_ranking():
    d = pd.concat([
        bars("LIQ", [50.0] * 30, volume=500_000),      # most liquid, passes
        bars("MID", [50.0] * 30, volume=100_000),      # passes
        bars("PENNY", [5.0] * 30, highs=[6.0] * 30, lows=[4.0] * 30, volume=10_000_000),
        bars("THIN", [50.0] * 30, volume=1_000),       # dollar volume too low
        bars("QUIET", [50.0] * 30, highs=[50.1] * 30, lows=[49.9] * 30, volume=500_000),
        bars("NEW", [50.0] * 5),                       # not enough history
    ])
    as_of = datetime(2027, 1, 1, tzinfo=UTC)
    out = select_universe(d, as_of, CFG)
    assert list(out["symbol"]) == ["LIQ", "MID"]
    assert list(select_universe(d, as_of, CFG, n=1)["symbol"]) == ["LIQ"]


def test_eligible_assets_drops_funds_otc_inactive_and_odd_tickers():
    rows = [
        ("AAPL", "Apple Inc. Common Stock", "NASDAQ", "active", True),
        ("SPY", "SPDR S&P 500 ETF Trust", "ARCA", "active", True),
        ("WARR", "Foo Corp Warrant", "NASDAQ", "active", True),
        ("BRK.B", "Berkshire Hathaway Class B", "NYSE", "active", True),
        ("OTCX", "Otc Corp", "OTC", "active", True),
        ("OLD", "Old Corp", "NYSE", "inactive", False),
        ("NOTR", "Halted Corp", "NYSE", "active", False),
        ("JPM", "JPMorgan Chase & Co. Common Stock", "NYSE", "active", True),
    ]
    a = pd.DataFrame(rows, columns=["symbol", "name", "exchange", "status", "tradable"])
    assert eligible_assets(a) == ["AAPL", "JPM"]


def test_membership_equals_select_universe_at_the_previous_close():
    from signalplat.engines.universe import membership

    cfgm = {**CFG, "min_atr_pct": 0.02}
    d = pd.concat([
        bars("LIQ", [50.0] * 40, volume=500_000),
        bars("PENNY", [5.0] * 40, highs=[6.0] * 40, lows=[4.0] * 40, volume=10_000_000),
        bars("QUIET", [50.0] * 40, highs=[50.1] * 40, lows=[49.9] * 40, volume=500_000),
    ])
    d["raw_close"] = d["close"]
    m = membership(d, cfgm).set_index(["symbol", "day"])["member"]
    for sym in ("LIQ", "PENNY", "QUIET"):
        g = d[d["symbol"] == sym].reset_index(drop=True)
        for i in (3, 14, 15, 25, 39):  # before and after the 15-bar warm-up
            prior = g.iloc[i - 1]["available_at"]  # information available before bar i
            chosen = set(select_universe(d, prior.to_pydatetime(), cfgm)["symbol"])
            day = g.iloc[i]["timestamp"].tz_convert("America/New_York").date()
            assert bool(m[(sym, day)]) == (sym in chosen), (sym, i)
    assert m[("LIQ", d[d.symbol == "LIQ"].iloc[39]["timestamp"]
              .tz_convert("America/New_York").date())]


def test_membership_price_rule_uses_the_price_visible_at_the_time():
    from signalplat.engines.universe import membership

    # $8 raw before a 1-for-10 reverse split: adjusted history says $80, but it was an $8 stock
    adj = bars("AAA", [80.0] * 40, volume=500_000)
    adj["raw_close"] = [8.0] * 30 + [80.0] * 10
    m = membership(adj, CFG).set_index("day")["member"]
    days = adj["timestamp"].dt.tz_convert("America/New_York").dt.date
    assert not m[days.iloc[25]]       # still an $8 stock then
    assert m[days.iloc[38]]           # an $80 stock after the split


def test_max_price_caps_both_the_selection_and_the_membership():
    from signalplat.engines.universe import membership

    cfg = {**CFG, "max_price": 60.0}
    d = pd.concat([bars("CHEAP", [50.0] * 40, volume=500_000),
                   bars("DEAR", [500.0] * 40, highs=[520.0] * 40, lows=[480.0] * 40,
                        volume=50_000)])
    d["raw_close"] = d["close"]
    as_of = datetime(2027, 1, 1, tzinfo=UTC)
    assert list(select_universe(d, as_of, cfg)["symbol"]) == ["CHEAP"]
    assert sorted(select_universe(d, as_of, CFG)["symbol"]) == ["CHEAP", "DEAR"]  # no cap set
    m = membership(d, cfg)
    last = m.groupby("symbol")["member"].last()
    assert bool(last["CHEAP"]) and not bool(last["DEAR"])
