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


def _history(symbol, days, minutes_per_day, volume_per_minute=100):
    """Liquid-looking daily bars plus regular-session minute bars for each day."""
    minute, daily = [], []
    for d in days:
        m = _bars(symbol, f"{d:%Y-%m-%d}", n=minutes_per_day, volume=volume_per_minute)
        minute.append(m)
        ts = pd.Timestamp(f"{d:%Y-%m-%d} 00:00", tz="America/New_York").tz_convert("UTC")
        daily.append(pd.DataFrame({
            "symbol": [symbol], "timestamp": [ts], "open": 50.0, "high": 52.0, "low": 48.0,
            "close": 50.0, "volume": float(m["volume"].sum()), "vwap": 50.0, "trade_count": 5,
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
    df = pd.concat(frames, ignore_index=True)
    ParquetBars(tmp_path).write_minute(df, Feed.SIP)
    ParquetBars(tmp_path).write_minute(df.iloc[:100], Feed.SIP)  # overlapping re-fetch
    got = ParquetBars(tmp_path).minute_summary(
        ["AAA", "BBB"], datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC),
        Feed.SIP)
    want = summarize_minutes(df)
    key = ["symbol", "day"]
    got = got.sort_values(key).reset_index(drop=True)
    want = want.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False, check_exact=False)
    assert ParquetBars(tmp_path).minute_summary([], datetime(2026, 9, 1, tzinfo=UTC),
                                                datetime(2026, 11, 1, tzinfo=UTC), Feed.SIP).empty
    assert ParquetBars(tmp_path).minute_summary(["AAA"], datetime(2026, 9, 1, tzinfo=UTC),
                                                datetime(2026, 11, 1, tzinfo=UTC), Feed.IEX).empty
