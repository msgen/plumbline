import json
from datetime import UTC, datetime

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


def test_e0_end_to_end_pass_and_fail(tmp_path):
    bars, store = ParquetBars(tmp_path), DatasetStore(tmp_path)
    bars.write_minute(_bars("AAA", "2026-10-01"), Feed.SIP)
    bars.write_daily(_daily("AAA", "2026-10-01", 39000))
    assets = pd.DataFrame({"symbol": ["DAA", "DBB"], "name": "n", "exchange": "NYSE",
                           "status": "inactive", "tradable": False, "asset_class": "us_equity"})
    store.write_assets(assets)
    bars.write_daily(pd.concat([_daily("DAA", "2026-10-01", 1), _daily("DBB", "2026-10-01", 1)]))
    cfg = tmp_path / "e0.yaml"
    cfg.write_text("gate: {delisted_present_min: 0.9, missing_regular_minutes_max: 0.005,"
                   " daily_vs_minute_volume_tol: 0.02}\ndelisted_sample: 2\nseed: 1\n"
                   "delisted_start: 2016-01-01\n")
    mgr = ExperimentManager(bars, store, ExperimentStore(tmp_path),
                            FixedClock(datetime(2026, 10, 5, 12, 0, tzinfo=UTC)))
    s, e = datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 10, 5, tzinfo=UTC)
    res = mgr.run_e0(cfg, ["AAA"], s, e)
    assert res["passed"], res["verdict"]
    assert (tmp_path / "experiments" / res["run_id"] / "report.md").exists()
    run = json.loads((tmp_path / "experiments" / res["run_id"] / "run.json").read_text())
    assert run["dataset_hashes"]["bars_1m"] and run["config_hash"]

    # drop 100 minutes of SIP data for a second symbol: the missing-minutes gate must fail
    bars.write_minute(_bars("BBB", "2026-10-01", n=290), Feed.SIP)
    bars.write_daily(_daily("BBB", "2026-10-01", 29000))
    res = mgr.run_e0(cfg, ["AAA", "BBB"], s, e)
    assert not res["passed"]
    assert res["verdict"]["missing_regular_minutes"]["status"] == "fail"
