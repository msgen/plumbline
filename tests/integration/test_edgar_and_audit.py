import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from signalplat.accessors.bars_parquet import ParquetBars
from signalplat.accessors.dataset_store import DatasetStore
from signalplat.accessors.experiment_store import ExperimentStore
from signalplat.accessors.filings_edgar import EdgarFilings
from signalplat.contracts.types import Feed
from signalplat.managers.experiment import ExperimentManager
from signalplat.utilities.clock import FixedClock
from signalplat.utilities.http import JsonHttp
from signalplat.utilities.ratelimit import RateLimiter

ROOT = Path(__file__).resolve().parents[2]
TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}}
RECENT = {
    "accessionNumber": ["a1", "a2", "a3", "a4"],
    "form": ["8-K", "10-Q", "424B5", "S-3"],
    "items": ["2.02,9.01", "", "", ""],
    "acceptanceDateTime": ["2026-08-01T20:15:30.000Z", "2026-08-02T12:00:00.000Z",
                           "2026-08-03T13:00:00.000Z", "2024-01-01T13:00:00.000Z"],
    "primaryDocument": ["x.htm", "y.htm", "z.htm", "w.htm"],
}
OLDER = {**{k: [v[0]] for k, v in RECENT.items()}, "accessionNumber": ["old1"],
         "form": ["S-1"], "acceptanceDateTime": ["2026-07-01T12:00:00.000Z"]}


def http(routes):
    def transport(url, headers):
        for path, body in routes.items():
            if path in url:
                return 200, json.dumps(body).encode()
        return 404, b"{}"
    return JsonHttp("https://x", {}, transport=transport, sleep=lambda s: None)


def test_edgar_filters_forms_window_and_follows_older_pages():
    www = http({"company_tickers": TICKERS})
    data = http({
        "CIK0000320193.json": {"filings": {
            "recent": RECENT, "files": [{"name": "CIK0000320193-submissions-001.json"}]}},
        "submissions-001": OLDER,
    })
    acc = EdgarFilings(www, data, RateLimiter(1000, 1.0))
    df = acc.filings(["AAPL", "GONE"], datetime(2026, 6, 1, tzinfo=UTC),
                     datetime(2026, 9, 1, tzinfo=UTC))
    assert acc.unmapped == {"GONE"}  # surfaced, not silently dropped
    assert sorted(df["accession"]) == ["a1", "a3", "old1"]  # 10-Q and out-of-window dropped
    row = df[df["accession"] == "a1"].iloc[0]
    assert row["items"] == "2.02,9.01" and row["available_at"] == row["accepted_at"]


def test_edgar_fails_loudly_on_unexpected_shape():
    www = http({"company_tickers": TICKERS})
    bad = {"filings": {"recent": {"accessionNumber": ["a"], "form": ["8-K"]}}}
    acc = EdgarFilings(www, http({"CIK0000320193.json": bad}), RateLimiter(1000, 1.0))
    with pytest.raises(KeyError, match="acceptanceDateTime"):
        acc.filings(["AAPL"], datetime(2026, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC))


def _bars(symbol, day, n=390, volume=100):
    ts = pd.date_range(f"{day} 09:30", periods=n, freq="min", tz="America/New_York")
    ts = ts.tz_convert("UTC")
    return pd.DataFrame({
        "symbol": symbol, "timestamp": ts, "open": 10.0, "high": 11.0, "low": 9.0,
        "close": 10.0, "volume": volume, "vwap": 10.0, "trade_count": 5,
        "available_at": ts + pd.Timedelta("1min")})


def _daily(symbol, day, volume):
    ts = pd.Timestamp(f"{day} 00:00", tz="America/New_York").tz_convert("UTC")
    return pd.DataFrame({
        "symbol": [symbol], "timestamp": [ts], "open": 10.0, "high": 11.0, "low": 9.0,
        "close": 10.0, "volume": volume, "vwap": 10.0, "trade_count": 5,
        "available_at": [ts + pd.Timedelta("1D")]})


def _history(symbol, days, minutes_per_day, volume_per_minute=100, price=50.0):
    """Liquid-looking daily bars plus regular-session minute bars for each day."""
    minute, daily = [], []
    for d in days:
        m = _bars(symbol, f"{d:%Y-%m-%d}", n=minutes_per_day, volume=volume_per_minute)
        minute.append(m)
        ts = pd.Timestamp(f"{d:%Y-%m-%d} 00:00", tz="America/New_York").tz_convert("UTC")
        daily.append(pd.DataFrame({
            "symbol": [symbol], "timestamp": [ts], "open": price, "high": price * 1.04,
            "low": price * 0.96, "close": price, "volume": float(m["volume"].sum()),
            "vwap": 50.0, "trade_count": 5,
            "available_at": [ts + pd.Timedelta("1D")]}))
    return pd.concat(minute, ignore_index=True), pd.concat(daily, ignore_index=True)


def test_e0_end_to_end_gates_apply_to_universe_days_only(tmp_path):
    bars, store = ParquetBars(tmp_path), DatasetStore(tmp_path)
    days = pd.bdate_range("2026-08-03", periods=30)
    m, d = _history("AAA", days, 390)
    bars.write_minute(m, Feed.SIP)
    bars.write_daily(d)
    assets = pd.DataFrame({"symbol": ["DAA", "DBB"], "name": "n", "exchange": "NYSE",
                           "status": "inactive", "tradable": False, "asset_class": "us_equity"})
    store.write_assets(assets)
    bars.write_daily(pd.concat([_daily("DAA", "2026-10-01", 1), _daily("DBB", "2026-10-01", 1)]))
    universe = tmp_path / "universe.yaml"
    universe.write_text("min_price: 10\nmin_dollar_volume_20d: 1000\nmin_atr_pct: 0.01\n")
    cfg = tmp_path / "e0.yaml"
    cfg.write_text("gate: {delisted_present_min: 0.9, missing_regular_minutes_max: 0.005,"
                   " daily_vs_minute_volume_tol: 0.02}\ndelisted_sample: 2\nseed: 1\n"
                   "delisted_start: 2016-01-01\n"
                   f"universe_config: {universe}\n")
    mgr = ExperimentManager(bars, store, ExperimentStore(tmp_path),
                            FixedClock(datetime(2026, 10, 5, 12, 0, tzinfo=UTC)))
    s, e = datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 10, 5, tzinfo=UTC)
    res = mgr.run_e0(cfg, ["AAA"], s, e)
    assert res["passed"], res["verdict"]
    assert res["metrics"]["missing_minutes"]["scope"] == "universe days"
    report = tmp_path / "experiments" / res["run_id"]
    assert (report / "report.md").exists()
    run = json.loads((report / "run.json").read_text())
    assert run["dataset_hashes"]["bars_raw_1m"] and run["config_hash"]

    # BBB loses 100 minutes a day once it has become a universe member: the gate must fail
    m2, d2 = _history("BBB", days, 290)
    bars.write_minute(m2, Feed.SIP)
    bars.write_daily(d2)
    res = mgr.run_e0(cfg, ["AAA", "BBB"], s, e)
    assert not res["passed"]
    assert res["verdict"]["missing_regular_minutes"]["status"] == "fail"

    # a thin stock that never qualifies (price under $10) cannot fail the gate
    thin_m, thin_d = _history("THIN", days, 20)
    thin_d[["open", "high", "low", "close"]] = [[5.0, 5.2, 4.8, 5.0]] * len(thin_d)
    bars.write_minute(thin_m, Feed.SIP)
    bars.write_daily(thin_d)
    res = mgr.run_e0(cfg, ["AAA", "THIN"], s, e)
    assert res["passed"], res["verdict"]
    assert res["metrics"]["missing_minutes"]["fraction_all"] > 0.05  # shown, not gated


def test_duckdb_minute_summary_matches_the_pandas_reference(tmp_path):
    from signalplat.engines.quality import summarize_minutes

    frames = [
        _bars("AAA", "2026-10-01"),
        _bars("AAA", "2026-10-02", n=210),                     # half day
        _bars("BBB", "2026-10-01").iloc[39:],                  # gap at the open
        _bars("BBB", "2026-10-02").assign(volume=7.0),
    ]
    pre = _bars("AAA", "2026-10-05")
    pre["timestamp"] = pre["timestamp"] - pd.Timedelta(hours=7)  # starts at 02:30 New York
    frames.append(pre)
    frames.append(_bars("AAA", "2026-11-27", n=330))  # early close: regular session ends 13:00
    # a New York day that straddles two UTC month directories: the 20:00 bar lands in October
    frames.append(_bars("AAA", "2026-09-30"))
    late = _bars("AAA", "2026-09-30", n=6)
    late["timestamp"] = pd.date_range(
        "2026-09-30 19:55", periods=6, freq="min", tz="America/New_York").tz_convert("UTC")
    frames.append(late)
    df = pd.concat(frames, ignore_index=True)
    ParquetBars(tmp_path).write_minute(df, Feed.SIP)
    ParquetBars(tmp_path).write_minute(df.iloc[:100], Feed.SIP)  # overlapping re-fetch
    got = ParquetBars(tmp_path).minute_summary(
        ["AAA", "BBB"], datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 12, 31, tzinfo=UTC),
        Feed.SIP)
    want = summarize_minutes(df)
    key = ["symbol", "day"]
    got = got.sort_values(key).reset_index(drop=True)
    want = want.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False, check_exact=False)
    assert ParquetBars(tmp_path).minute_summary([], datetime(2026, 9, 1, tzinfo=UTC),
                                                datetime(2026, 12, 31, tzinfo=UTC), Feed.SIP).empty
    assert ParquetBars(tmp_path).minute_summary(["AAA"], datetime(2026, 9, 1, tzinfo=UTC),
                                                datetime(2026, 12, 31, tzinfo=UTC), Feed.IEX).empty


def test_reviewed_jumps_clear_the_review_status(tmp_path):
    bars, store = ParquetBars(tmp_path), DatasetStore(tmp_path)
    days = pd.bdate_range("2026-08-03", periods=30)
    m, d = _history("AAA", days, 390)
    d.loc[d.index[-1], ["open", "high", "low", "close"]] = [100.0, 104.0, 96.0, 100.0]  # +100%
    bars.write_minute(m, Feed.SIP)
    bars.write_daily(d)
    store.write_assets(pd.DataFrame({
        "symbol": ["DAA"], "name": "n", "exchange": "NYSE", "status": "inactive",
        "tradable": False, "asset_class": "us_equity"}))
    universe = tmp_path / "universe.yaml"
    universe.write_text("min_price: 10\nmin_dollar_volume_20d: 1000\nmin_atr_pct: 0.01\n")
    day = str(days[-1].date())

    def run(reviewed):
        cfg = tmp_path / "e0.yaml"
        cfg.write_text(
            "gate: {delisted_present_min: 0.0, missing_regular_minutes_max: 0.005,"
            " daily_vs_minute_volume_tol: 0.5}\ndelisted_sample: 1\nseed: 1\n"
            f"delisted_start: 2016-01-01\nuniverse_config: {universe}\n"
            f"reviewed_jumps: {reviewed}\n")
        mgr = ExperimentManager(bars, store, ExperimentStore(tmp_path),
                                FixedClock(datetime(2026, 10, 5, 12, 0, tzinfo=UTC)))
        return mgr.run_e0(cfg, ["AAA"], datetime(2026, 8, 1, tzinfo=UTC),
                          datetime(2026, 10, 5, tzinfo=UTC))

    assert run("[]")["verdict"]["price_jumps"]["status"] == "review"
    res = run(f'[{{symbol: AAA, day: "{day}", note: "checked: earnings"}}]')
    assert res["verdict"]["price_jumps"]["status"] == "pass"
    assert res["metrics"]["price_jumps_reviewed"] == 1


def test_inspect_day_reports_counts_volume_and_gaps(tmp_path):
    bars, store = ParquetBars(tmp_path), DatasetStore(tmp_path)
    m = _bars("AAA", "2026-10-01")
    m = m[(m.index < 100) | (m.index >= 150)].reset_index(drop=True)  # 50 minutes missing
    bars.write_minute(m, Feed.SIP)
    bars.write_daily(_daily("AAA", "2026-10-01", float(m["volume"].sum())))
    mgr = ExperimentManager(bars, store, ExperimentStore(tmp_path),
                            FixedClock(datetime(2026, 10, 5, 12, 0, tzinfo=UTC)))
    info = mgr.inspect_day("AAA", datetime(2026, 10, 1, tzinfo=UTC))
    assert info["regular_minute_bars"] == 340 and info["first"] == "09:30"
    assert info["last"] == "15:59"
    assert info["longest_gaps_regular"] == [("11:10", 50)]
    assert info["minute_volume_regular"] == info["daily_volume"][0]
    assert mgr.inspect_day("ZZZ", datetime(2026, 10, 1, tzinfo=UTC))["minute_bars"] == 0


def test_expensive_stocks_stay_in_the_universe_but_not_in_the_completeness_gate(tmp_path):
    bars, store = ParquetBars(tmp_path), DatasetStore(tmp_path)
    days = pd.bdate_range("2026-08-03", periods=30)
    for sym, minutes, price in (("CHEAP", 390, 40.0), ("DEAR", 100, 500.0)):
        m, d = _history(sym, days, minutes, price=price)
        bars.write_minute(m, Feed.SIP)
        bars.write_daily(d)
    store.write_assets(pd.DataFrame({
        "symbol": ["DAA"], "name": "n", "exchange": "NYSE", "status": "inactive",
        "tradable": False, "asset_class": "us_equity"}))
    universe = tmp_path / "universe.yaml"
    universe.write_text("min_price: 10\nmin_dollar_volume_20d: 1000\nmin_atr_pct: 0.01\n"
                        "complete_bars_max_price: 100\n")
    cfg = tmp_path / "e0.yaml"
    cfg.write_text(
        "gate: {delisted_present_min: 0.0, missing_regular_minutes_max: 0.005,"
        " daily_vs_minute_volume_tol: 0.9}\ndelisted_sample: 1\nseed: 1\n"
        f"delisted_start: 2016-01-01\nuniverse_config: {universe}\n")
    mgr = ExperimentManager(bars, store, ExperimentStore(tmp_path),
                            FixedClock(datetime(2026, 10, 5, 12, 0, tzinfo=UTC)))
    res = mgr.run_e0(cfg, ["CHEAP", "DEAR"], datetime(2026, 8, 1, tzinfo=UTC),
                     datetime(2026, 10, 5, tzinfo=UTC))
    gate, full = res["metrics"]["missing_minutes"], res["metrics"]["missing_minutes_full"]
    assert res["verdict"]["missing_regular_minutes"]["status"] == "pass"
    assert gate["fraction"] == 0.0                       # cheap stock only
    assert full["fraction"] > 0.3                        # DEAR is in the universe, and gappy
    assert "250-1000" in full["by_price"]
    assert "priced below $100" in (tmp_path / "experiments" / res["run_id"] / "report.md"
                                   ).read_text()


def test_opening_volume_matches_a_pandas_calculation(tmp_path):
    df = pd.concat([_bars("AAA", "2026-10-01"), _bars("AAA", "2026-10-02")], ignore_index=True)
    df["volume"] = range(1, len(df) + 1)
    df["vwap"] = 10.0 + (df["volume"] % 7) / 10
    ParquetBars(tmp_path).write_minute(df, Feed.SIP)
    ParquetBars(tmp_path).write_minute(df.iloc[:50], Feed.SIP)  # overlapping re-fetch
    got = ParquetBars(tmp_path).opening_volume(
        ["AAA"], datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 12, 1, tzinfo=UTC), Feed.SIP,
        [575, 600])
    local = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.assign(day=local.dt.date, mod=local.dt.hour * 60 + local.dt.minute)
    for minute, lab in ((575, "0935"), (600, "1000")):
        sel = df[(df["mod"] >= 570) & (df["mod"] < minute)]
        want_v = sel.groupby("day")["volume"].sum()
        want_pv = (sel["vwap"] * sel["volume"]).groupby(sel["day"]).sum()
        g = got.set_index("day")
        assert list(g[f"v_{lab}"]) == list(want_v)
        assert all(abs(a - b) < 1e-6 for a, b in zip(g[f"pv_{lab}"], want_pv, strict=True))
    assert ParquetBars(tmp_path).opening_volume(
        ["AAA"], datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 12, 1, tzinfo=UTC), Feed.IEX,
        [575]).empty


def test_e1_end_to_end(tmp_path):
    bars, store = ParquetBars(tmp_path), DatasetStore(tmp_path)
    days = pd.bdate_range("2026-08-03", periods=30)
    for feed, per_minute in ((Feed.SIP, 100), (Feed.IEX, 10)):  # IEX sees a tenth of the volume
        m, d = _history("AAA", days, 390, volume_per_minute=per_minute)
        busy = m["timestamp"].dt.tz_convert("America/New_York").dt.date == days[-1].date()
        m.loc[busy, "volume"] *= 3        # the last day is three times as busy in both feeds
        bars.write_minute(m, feed)
        if feed is Feed.SIP:
            bars.write_daily(d)
    universe = tmp_path / "universe.yaml"
    universe.write_text("min_price: 10\nmin_dollar_volume_20d: 1000\nmin_atr_pct: 0.01\n")
    cfg = tmp_path / "e1.yaml"
    cfg.write_text(
        'times: ["09:35", "09:45", "10:00", "10:30"]\ntrailing_days: 20\nmin_history_days: 10\n'
        'rvol_flag: 2.5\nerror_time: "10:00"\nflag_time: "09:45"\n'
        "gate: {median_rvol_error: 0.15, flag_agreement: 0.9}\n"
        f"universe_config: {universe}\n")
    mgr = ExperimentManager(bars, store, ExperimentStore(tmp_path),
                            FixedClock(datetime(2026, 10, 5, 12, 0, tzinfo=UTC)))
    res = mgr.run_e1(cfg, ["AAA"], datetime(2026, 8, 1, tzinfo=UTC),
                     datetime(2026, 10, 5, tzinfo=UTC))
    assert res["passed"], res["verdict"]
    assert res["verdict"]["median_rvol_error_1000"]["value"] < 1e-9
    assert res["verdict"]["flag_agreement_0945"]["value"] == 1.0   # both feeds flag the busy day
    report = (tmp_path / "experiments" / res["run_id"] / "report.md").read_text()
    assert "IEX versus SIP" in report and "09:45" in report
