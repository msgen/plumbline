from datetime import UTC, datetime

import pandas as pd

from signalplat.engines import quality as q

NY = "America/New_York"


def minutes(symbol, day, start="09:30", n=390, volume=100):
    ts = pd.date_range(f"{day} {start}", periods=n, freq="min", tz=NY).tz_convert("UTC")
    return pd.DataFrame({"symbol": symbol, "timestamp": ts, "volume": volume, "close": 10.0})


def daily(symbol, day, close=10.0, volume=39000):
    ts = pd.Timestamp(f"{day} 00:00", tz=NY).tz_convert("UTC")
    return pd.DataFrame({"symbol": [symbol], "timestamp": [ts], "close": [close],
                         "volume": [volume]})


def test_missing_minutes_counts_gaps_and_ignores_half_days():
    full = minutes("AAA", "2026-10-01")
    gappy = minutes("BBB", "2026-10-01").iloc[39:]  # 39 minutes missing
    d = pd.concat([daily("AAA", "2026-10-01"), daily("BBB", "2026-10-01")])
    r = q.missing_regular_minutes(q.summarize_minutes(pd.concat([full, gappy])), d)
    assert r["missing"] == 39 and r["symbol_days"] == 2
    assert [w[:3] for w in r["worst"]] == [("BBB", "2026-10-01", 39)]  # only days with gaps
    assert abs(r["fraction"] - 39 / 780) < 1e-9
    # early close: market-wide span is 210 minutes, so nothing is missing
    half = pd.concat([minutes("AAA", "2026-11-27", n=210), minutes("BBB", "2026-11-27", n=210)])
    r = q.missing_regular_minutes(q.summarize_minutes(half), pd.concat(
        [daily("AAA", "2026-11-27"), daily("BBB", "2026-11-27")]))
    assert r["missing"] == 0


def test_day_with_no_minute_data_counts_fully_missing():
    r = q.missing_regular_minutes(q.summarize_minutes(minutes("AAA", "2026-10-01").iloc[0:0]),
                                  daily("AAA", "2026-10-01"))
    assert r["missing"] == 390 and r["fraction"] == 1.0


def test_duplicates_and_extended_hours():
    m = minutes("AAA", "2026-10-01", n=5)
    assert q.duplicate_count(pd.concat([m, m.iloc[:2]])) == 2
    assert q.duplicate_minute_rows(q.summarize_minutes(pd.concat([m, m.iloc[:2]]))) == 2
    assert q.outside_extended_hours(q.summarize_minutes(m)) == 0
    early = q.summarize_minutes(minutes("AAA", "2026-10-01", start="03:00", n=10))
    assert q.outside_extended_hours(early) == 10


def test_price_jumps():
    d = pd.concat([daily("AAA", "2026-10-01", 10), daily("AAA", "2026-10-02", 20),
                   daily("BBB", "2026-10-01", 10), daily("BBB", "2026-10-02", 11)])
    jumps = q.price_jumps(d)
    assert [(s, day) for s, day, _ in jumps] == [("AAA", "2026-10-02")]


def test_volume_consistency_picks_matching_convention():
    reg = minutes("AAA", "2026-10-01", volume=100)  # 39,000 regular
    pre = minutes("AAA", "2026-10-01", start="08:00", n=90, volume=100)  # 9,000 premarket
    m = pd.concat([pre, reg])
    m = q.summarize_minutes(m)
    r = q.volume_consistency(m, daily("AAA", "2026-10-01", volume=39000))
    assert r["convention"] == "regular" and r["median_rel_diff"] == 0
    r = q.volume_consistency(m, daily("AAA", "2026-10-01", volume=48000))
    assert r["convention"] == "all_hours" and r["median_rel_diff"] == 0
    assert q.volume_consistency(m.iloc[0:0], daily("AAA", "2026-10-01"))["symbol_days"] == 0


def test_delisted_sampling_is_seeded_and_excludes_otc():
    assets = pd.DataFrame({
        "symbol": [f"S{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(30)]
        + ["ACT", "OTCX", "464ESC045"],
        "status": ["inactive"] * 30 + ["active", "inactive", "inactive"],
        "exchange": ["NYSE"] * 30 + ["NYSE", "OTC", "NYSE"]})
    a = q.sample_delisted(assets, 10, seed=1)
    assert a == q.sample_delisted(assets, 10, seed=1) and len(a) == 10
    assert "ACT" not in a and "OTCX" not in a and "464ESC045" not in a
    cov = q.delisted_coverage(a, pd.DataFrame({"symbol": a[:9]}))
    assert cov["present"] == 9 and cov["missing"] == [a[9]] and cov["fraction"] == 0.9


def test_evaluate_e0_statuses_and_none_handling():
    gate = {"delisted_present_min": 0.9, "missing_regular_minutes_max": 0.005,
            "daily_vs_minute_volume_tol": 0.02}
    metrics = {"delisted": {"fraction": 0.8}, "missing_minutes": {"fraction": 0.001},
               "volume": {"median_rel_diff": None}, "duplicates": 0, "outside_hours": 0,
               "price_jumps": [("A", "2026-01-01", 0.5)]}
    v = q.evaluate_e0(metrics, gate)
    assert v["delisted_present"]["status"] == "fail"
    assert v["missing_regular_minutes"]["status"] == "pass"
    assert v["daily_vs_minute_volume"]["status"] == "n/a"
    assert v["price_jumps"]["status"] == "review"
    assert datetime(2026, 1, 1, tzinfo=UTC)


def test_filings_and_news_coverage():
    filings = pd.DataFrame({"symbol": ["AAA", "AAA", "BBB"]})
    c = q.filings_coverage(["AAA", "BBB", "CCC"], ["OLD1", "OLD2"], filings)
    assert c["universe"] == {"with_filings": 2, "total": 3, "fraction": 2 / 3}
    assert c["delisted_sample"]["with_filings"] == 0
    assert q.filings_coverage(["AAA"], [], None)["universe"]["with_filings"] == 0
    news = pd.DataFrame({"id": ["1", "2", "3"], "symbols": [["AAA"], ["AAA", "BBB"], []]})
    n = q.news_coverage(news, ["AAA", "BBB", "CCC"], days=2)
    assert n == {"articles": 3, "per_day": 1.5, "symbols_with_news": 2, "symbols": 3}
    assert q.news_coverage(None, ["AAA"], 5)["articles"] == 0


def test_the_2000_minute_is_a_boundary_not_an_outside_bar():
    ts = pd.to_datetime(["2026-10-01 19:59", "2026-10-01 20:00", "2026-10-01 20:01",
                         "2026-10-01 03:59"]).tz_localize(NY).tz_convert("UTC")
    m = pd.DataFrame({"symbol": "AAA", "timestamp": ts, "volume": 1.0, "trade_count": 1})
    s = q.summarize_minutes(m)
    assert q.boundary_bars(s) == 1
    assert q.outside_extended_hours(s) == 2  # 20:01 and 03:59


def test_missing_minutes_reported_by_trade_density():
    # a busy stock (many trades per minute) with a full day, a sparse one missing most minutes
    busy = minutes("BUSY", "2026-10-01").assign(trade_count=20)
    sparse = minutes("SPARSE", "2026-10-01").iloc[:60].assign(trade_count=1)
    d = pd.concat([daily("BUSY", "2026-10-01"), daily("SPARSE", "2026-10-01")])
    r = q.missing_regular_minutes(q.summarize_minutes(pd.concat([busy, sparse])), d)
    dens = r["by_trades_per_minute"]
    assert dens[">10"]["missing_fraction"] == 0.0
    assert dens["<0.5"]["missing_fraction"] == 330 / 390


def test_price_jumps_only_on_universe_days_when_members_given():
    d = pd.concat([daily("AAA", "2026-10-01", 10), daily("AAA", "2026-10-02", 20)],
                  ignore_index=True)
    day = pd.Timestamp("2026-10-02").date()
    yes = pd.DataFrame({"symbol": ["AAA"], "day": [day], "member": [True]})
    no = pd.DataFrame({"symbol": ["AAA"], "day": [day], "member": [False]})
    assert len(q.price_jumps(d, members=yes)) == 1
    assert q.price_jumps(d, members=no) == []
    assert len(q.price_jumps(d)) == 1


def test_early_close_day_uses_the_13_00_session_end():
    # 2026-11-27 closes at 13:00. Extended-hours prints until 15:00 must not stretch the session.
    a = minutes("AAA", "2026-11-27", n=330)  # 09:30 to 15:00
    b = minutes("BBB", "2026-11-27", n=210)  # 09:30 to 13:00
    d = pd.concat([daily("AAA", "2026-11-27"), daily("BBB", "2026-11-27")])
    s = q.summarize_minutes(pd.concat([a, b]))
    assert list(s.sort_values("symbol")["reg_minutes"]) == [210, 210]
    r = q.missing_regular_minutes(s, d)
    assert r["missing"] == 0 and r["expected"] == 420


def test_volume_check_tells_sparse_trading_from_lost_trades():
    # same 300 missing minutes: in one case the bars still add up to the daily volume
    kept = minutes("KEEP", "2026-10-01", volume=100).iloc[:90]
    lost = minutes("LOST", "2026-10-01", volume=100).iloc[:90]
    full = minutes("FULL", "2026-10-01", volume=100)
    d = pd.concat([daily("KEEP", "2026-10-01", volume=9000),    # matches the 90 bars
                   daily("LOST", "2026-10-01", volume=39000),   # as if all 390 had traded
                   daily("FULL", "2026-10-01", volume=39000)])
    r = q.missing_regular_minutes(q.summarize_minutes(pd.concat([kept, lost, full])), d)
    assert r["gappy_days_volume"]["days"] == 2
    assert r["gappy_days_volume"]["share_within_2pct"] == 0.5
    by_symbol = {w[0]: w[3] for w in r["worst"]}
    assert by_symbol["KEEP"] == 0.0 and by_symbol["LOST"] > 0.5
