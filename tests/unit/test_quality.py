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
    r = q.missing_regular_minutes(pd.concat([full, gappy]), d)
    assert r["missing"] == 39 and r["symbol_days"] == 2
    assert r["worst"] == [("BBB", "2026-10-01", 39)]  # complete days are not listed
    assert abs(r["fraction"] - 39 / 780) < 1e-9
    # early close: market-wide span is 210 minutes, so nothing is missing
    half = pd.concat([minutes("AAA", "2026-11-27", n=210), minutes("BBB", "2026-11-27", n=210)])
    r = q.missing_regular_minutes(half, pd.concat([daily("AAA", "2026-11-27"),
                                                   daily("BBB", "2026-11-27")]))
    assert r["missing"] == 0


def test_day_with_no_minute_data_counts_fully_missing():
    r = q.missing_regular_minutes(minutes("AAA", "2026-10-01").iloc[0:0],
                                  daily("AAA", "2026-10-01"))
    assert r["missing"] == 390 and r["fraction"] == 1.0


def test_duplicates_and_extended_hours():
    m = minutes("AAA", "2026-10-01", n=5)
    assert q.duplicate_count(pd.concat([m, m.iloc[:2]])) == 2
    assert q.outside_extended_hours(m) == 0
    assert q.outside_extended_hours(minutes("AAA", "2026-10-01", start="03:00", n=10)) == 10


def test_price_jumps():
    d = pd.concat([daily("AAA", "2026-10-01", 10), daily("AAA", "2026-10-02", 20),
                   daily("BBB", "2026-10-01", 10), daily("BBB", "2026-10-02", 11)])
    jumps = q.price_jumps(d)
    assert [(s, day) for s, day, _ in jumps] == [("AAA", "2026-10-02")]


def test_volume_consistency_picks_matching_convention():
    reg = minutes("AAA", "2026-10-01", volume=100)  # 39,000 regular
    pre = minutes("AAA", "2026-10-01", start="08:00", n=90, volume=100)  # 9,000 premarket
    m = pd.concat([pre, reg])
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
