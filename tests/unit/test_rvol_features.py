import pandas as pd

from signalplat.engines.features import rvol_grids

DAYS = [d.date() for d in pd.bdate_range("2026-08-03", periods=30)]


def minutes(per_minute_by_day, symbol="AAA", n=60):
    rows = []
    for day, v in zip(DAYS, per_minute_by_day, strict=True):
        for m in range(570, 570 + n):
            rows.append((symbol, day, m, v))
    return pd.DataFrame(rows, columns=["symbol", "day", "mod", "volume"])


def at(out, day, mod):
    r = out[(out["day"] == day) & (out["mod"] == mod)].iloc[0]
    return r["rvol_sip"], r["rvol_est"]


def test_rvol_matches_hand_calculation_and_scales_iex():
    sip = minutes([100.0] * 29 + [300.0])         # last day three times busier
    iex = minutes([10.0] * 29 + [25.0])           # IEX is a tenth, but only 2.5x busier today
    out = rvol_grids(sip, iex, until=630)
    s, e = at(out, DAYS[-1], 599)                 # 30 minutes in: cumulative 30 bars
    assert abs(s - 3.0) < 1e-12                    # 9000 / 3000
    assert abs(e - 2.5) < 1e-12                    # 750 * ratio 10 / 3000
    s0, e0 = at(out, DAYS[-1], 570)               # first minute
    assert abs(s0 - 3.0) < 1e-12 and abs(e0 - 2.5) < 1e-12


def test_early_days_have_no_rvol_until_there_is_history():
    out = rvol_grids(minutes([100.0] * 30), minutes([10.0] * 30), until=600)
    first = out[out["day"] == DAYS[2]]
    assert first["rvol_sip"].isna().all() and first["rvol_est"].isna().all()
    later = out[out["day"] == DAYS[15]]
    assert (later["rvol_sip"] == 1.0).all()


def test_only_earlier_days_enter_the_baseline():
    base = [100.0] * 30
    clean = rvol_grids(minutes(base), minutes([10.0] * 30), until=600)
    poisoned = rvol_grids(minutes(base[:20] + [9e9] * 10), minutes([10.0] * 20 + [1.0] * 10),
                          until=600)
    a = clean[clean["day"] == DAYS[19]].reset_index(drop=True)
    b = poisoned[poisoned["day"] == DAYS[19]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_missing_minutes_and_a_missing_iex_symbol_do_not_break_it():
    sip = minutes([100.0] * 30)
    sip = sip[sip["mod"] != 580]                  # a minute with no trades
    out = rvol_grids(sip, sip.iloc[0:0], until=600)
    day = out[(out["day"] == DAYS[-1]) & (out["mod"] == 590)].iloc[0]
    assert day["rvol_sip"] == 1.0                 # the gap is zero volume in baseline too
    assert pd.isna(day["rvol_est"])               # no IEX data, no estimate
    assert rvol_grids(sip.iloc[0:0], sip.iloc[0:0]).empty
