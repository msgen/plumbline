from datetime import date

import pandas as pd

from signalplat.engines.feeds import evaluate_e1, rvol_study

MIN = [575, 585]  # 09:35 and 09:45
DAYS = [d.date() for d in pd.bdate_range("2026-08-03", periods=30)]


def feed(volumes, vwap=10.0):
    return pd.DataFrame({
        "symbol": "AAA", "day": DAYS,
        "v_0935": volumes, "pv_0935": [v * vwap for v in volumes],
        "v_0945": volumes, "pv_0945": [v * vwap for v in volumes]})


def members(days, price=40.0):
    return pd.DataFrame({"symbol": "AAA", "day": days, "member": True, "price": price})


def test_rvol_error_flags_and_vwap_match_hand_calculation():
    sip = [1000.0] * 29 + [3000.0]
    iex = [100.0] * 29 + [250.0]   # trailing ratio 10, so the estimate is 2.5 against a true 3.0
    study = rvol_study(feed(sip), feed(iex, vwap=10.1), members([DAYS[-1]]), MIN)
    s = study["0935"]
    assert s["n"] == 1
    assert abs(s["median_rvol_error"] - (0.5 / 3.0)) < 1e-12
    assert abs(s["median_vwap_bps"] - 100.0) < 1e-6          # 10.1 against 10.0
    assert s["flag_agreement_flagged"] == 1.0                 # 2.5 and 3.0 both reach 2.5
    assert s["flag_precision"] == 1.0 and s["flag_recall"] == 1.0


def test_flag_agreement_counts_only_days_either_feed_flags():
    sip = [1000.0] * 29 + [3000.0]
    iex = [100.0] * 29 + [200.0]   # estimate 2.0: below the flag while SIP says 3.0
    s = rvol_study(feed(sip), feed(iex), members([DAYS[-1]]), MIN)["0935"]
    assert s["flag_agreement_all"] == 0.0 and s["flag_agreement_flagged"] == 0.0
    assert s["flag_recall"] == 0.0
    quiet = rvol_study(feed([1000.0] * 30), feed([100.0] * 30), members(DAYS[-5:]), MIN)["0935"]
    assert quiet["flag_agreement_all"] == 1.0 and quiet["flag_agreement_flagged"] is None


def test_estimate_uses_only_earlier_days():
    day = DAYS[24]
    base_sip = [1000.0] * 30
    base_iex = [100.0] * 30
    clean = rvol_study(feed(base_sip), feed(base_iex), members([day]), MIN)
    poisoned_sip = base_sip[:25] + [9e9] * 5      # garbage after the day under test
    poisoned_iex = base_iex[:25] + [1.0] * 5
    poisoned = rvol_study(feed(poisoned_sip), feed(poisoned_iex), members([day]), MIN)
    assert clean == poisoned


def test_too_little_history_drops_the_day():
    s = rvol_study(feed([1000.0] * 30), feed([100.0] * 30), members([DAYS[3]]), MIN)["0935"]
    assert s["n"] == 0


def test_error_by_price_bucket_and_gate_lines():
    sip = [1000.0] * 29 + [3000.0]
    iex = [100.0] * 29 + [250.0]
    s = rvol_study(feed(sip), feed(iex), members([DAYS[-1]], price=400.0), MIN)
    assert list(s["0935"]["median_rvol_error_by_price"]) == ["250-1000"]
    gate = {"median_rvol_error": 0.15, "flag_agreement": 0.9}
    v = evaluate_e1(s, gate, "0935", "0945")
    assert v["median_rvol_error_0935"]["status"] == "fail"      # 0.167 is not under 0.15
    assert v["flag_agreement_0945"]["status"] == "pass"
    assert evaluate_e1({}, gate, "0935", "0945")["flag_agreement_0945"]["status"] == "n/a"
    assert date(2026, 8, 3) == DAYS[0]


def test_funnel_counts_where_symbol_days_drop_out():
    sip = feed([1000.0] * 30)
    iex = feed([100.0] * 30).iloc[:20]            # IEX covers only the first 20 days
    study = rvol_study(sip, iex, members(DAYS[10:25]), MIN)
    f = study["funnel"]
    assert (f["sip_symbol_days"], f["iex_symbol_days"], f["both_feeds"]) == (30, 20, 20)
    assert f["both_feeds_and_universe"] == 10 and f["universe_days_in_period"] == 15
    assert f["usable_by_time"]["0935"] == 10
