import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from signalplat.accessors.bars_alpaca import AlpacaBars
from signalplat.accessors.bars_parquet import ParquetBars
from signalplat.accessors.dataset_store import DatasetStore
from signalplat.accessors.news_alpaca import AlpacaNews
from signalplat.accessors.reference_alpaca import AlpacaReference
from signalplat.contracts.types import Feed
from signalplat.managers.ingestion import IngestionManager, month_starts
from signalplat.utilities.clock import FixedClock
from signalplat.utilities.env import load_env, require
from signalplat.utilities.http import HttpError, JsonHttp
from signalplat.utilities.ratelimit import RateLimiter

NOW = datetime(2026, 10, 5, 15, 0, tzinfo=UTC)


class FakeTransport:
    """Replays canned vendor responses keyed by path; records every URL requested."""

    def __init__(self, routes):
        self.routes, self.urls = routes, []

    def __call__(self, url, headers):
        self.urls.append(url)
        for path, responses in self.routes.items():
            if path in url:
                status, body = responses.pop(0) if len(responses) > 1 else responses[0]
                return status, json.dumps(body).encode()
        return 404, b"{}"


def bar(t, c=10.0):
    return {"t": t, "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 100, "vw": c, "n": 5}


def make_http(routes):
    tr = FakeTransport(routes)
    return JsonHttp("https://x", {}, transport=tr, sleep=lambda s: None), tr


def test_bars_paginate_clamp_end_and_set_available_at():
    http, tr = make_http({"/v2/stocks/bars": [
        (200, {"bars": {"AAPL": [bar("2026-10-01T13:30:00Z")]}, "next_page_token": "p2"}),
        (200, {"bars": {"AAPL": [bar("2026-10-01T13:31:00Z")],
                        "MSFT": [bar("2026-10-01T13:30:00Z")]},
               "next_page_token": None}),
    ]})
    acc = AlpacaBars(http, FixedClock(NOW), RateLimiter(1000))
    df = acc.minute_bars(["AAPL", "MSFT"], NOW - timedelta(days=5), NOW, Feed.SIP)
    assert len(df) == 3 and list(df["symbol"]) == ["AAPL", "AAPL", "MSFT"]
    assert (df["available_at"] - df["timestamp"] == timedelta(minutes=1)).all()
    assert len(tr.urls) == 2 and "page_token=p2" in tr.urls[1]
    clamped = (NOW - timedelta(minutes=15)).isoformat().replace("+", "%2B").replace(":", "%3A")
    assert f"end={clamped}" in tr.urls[0]
    assert "feed=sip" in tr.urls[0] and "adjustment=split" in tr.urls[0]


def test_bars_empty_response_and_iex_not_clamped():
    http, tr = make_http({"/v2/stocks/bars": [(200, {"bars": None, "next_page_token": None})]})
    acc = AlpacaBars(http, FixedClock(NOW), RateLimiter(1000))
    assert acc.minute_bars(["AAPL"], NOW - timedelta(days=1), NOW, Feed.IEX).empty
    assert acc.minute_bars([], NOW - timedelta(days=1), NOW, Feed.IEX).empty
    assert len(tr.urls) == 1 and "feed=iex" in tr.urls[0]


def test_http_retries_then_raises():
    http, tr = make_http({"/a": [(429, {}), (500, {}), (200, {"ok": 1})]})
    assert http.get("/a") == {"ok": 1} and len(tr.urls) == 3
    http, _ = make_http({"/a": [(403, {})]})
    with pytest.raises(HttpError):
        http.get("/a")


def test_reference_merges_active_and_inactive():
    http, _ = make_http({"/v2/assets": [
        (200, [{"symbol": "AAPL", "name": "Apple", "exchange": "NASDAQ", "status": "active",
                "tradable": True, "class": "us_equity"}]),
        (200, [{"symbol": "OLD", "name": "Old", "exchange": "NYSE", "status": "inactive",
                "tradable": False, "class": "us_equity"}]),
    ]})
    df = AlpacaReference(http).assets()
    assert set(df["symbol"]) == {"AAPL", "OLD"} and set(df["status"]) == {"active", "inactive"}


def test_news_pagination_dedup_and_lag():
    art = {"id": 1, "headline": "h", "summary": "s", "symbols": ["AAPL"],
           "created_at": "2026-10-01T14:00:00Z", "updated_at": "2026-10-01T14:00:05Z"}
    http, _ = make_http({"/v1beta1/news": [
        (200, {"news": [art], "next_page_token": "n2"}),
        (200, {"news": [art, {**art, "id": 2}], "next_page_token": None}),
    ]})
    items = AlpacaNews(http, limiter=RateLimiter(1000)).news(["AAPL"], NOW - timedelta(days=9), NOW)
    assert sorted(i.id for i in items) == ["1", "2"]
    assert items[0].available_at - items[0].created_at == timedelta(seconds=60)


def test_parquet_roundtrip_dedupes_and_filters(tmp_path):
    http, _ = make_http({"/v2/stocks/bars": [
        (200, {"bars": {"AAPL": [bar("2026-09-30T13:30:00Z"), bar("2026-10-01T13:30:00Z")]}}),
    ]})
    df = AlpacaBars(http, FixedClock(NOW), RateLimiter(1000)).minute_bars(
        ["AAPL"], NOW - timedelta(days=9), NOW, Feed.IEX)
    store = ParquetBars(tmp_path)
    store.write_minute(df, Feed.IEX)
    store.write_minute(df, Feed.IEX)  # re-ingesting identical data adds no duplicates
    out = store.minute_bars(["AAPL"], datetime(2026, 9, 1, tzinfo=UTC), NOW, Feed.IEX)
    assert len(out) == 2
    sub = store.minute_bars(["AAPL"], datetime(2026, 10, 1, tzinfo=UTC), NOW, Feed.IEX)
    assert len(sub) == 1
    assert store.minute_bars([], NOW - timedelta(days=9), NOW, Feed.IEX).empty
    assert store.minute_bars(["AAPL"], NOW - timedelta(days=9), NOW, Feed.SIP).empty


def test_manager_resumes_from_manifest(tmp_path):
    http, tr = make_http({
        "/v2/stocks/bars": [(200, {"bars": {"AAPL": [bar("2026-08-03T13:30:00Z")]}})],
        "/v2/assets": [(200, [])],
    })
    clock = FixedClock(NOW)
    mgr = IngestionManager(
        AlpacaBars(http, clock, RateLimiter(1000)), AlpacaReference(http),
        AlpacaNews(http, limiter=RateLimiter(1000)), ParquetBars(tmp_path),
        DatasetStore(tmp_path), clock)
    s, e = datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC)
    assert mgr.ingest_minute(["AAPL"], s, e, Feed.IEX) == 1
    calls = len(tr.urls)
    assert mgr.ingest_minute(["AAPL"], s, e, Feed.IEX) == 0  # finished chunk skipped
    assert len(tr.urls) == calls
    # a chunk that ends inside the SIP delay window is never marked done
    recent = NOW - timedelta(minutes=5)
    mgr.ingest_minute(["AAPL"], NOW - timedelta(days=1), recent, Feed.SIP)
    assert not DatasetStore(tmp_path).is_done(
        f"1m|sip|{(NOW - timedelta(days=1)):%F}|{recent:%F}|AAPL")


def test_month_starts_splits_on_calendar_months():
    w = month_starts(datetime(2025, 12, 15, tzinfo=UTC), datetime(2026, 2, 10, tzinfo=UTC))
    assert [(a.month, b.month, b.day) for a, b in w] == [(12, 1, 1), (1, 2, 1), (2, 2, 10)]


def test_env_loader(tmp_path):
    f = tmp_path / ".env"
    f.write_text('# c\nA=1\nB="two words"\nC=\n')
    env = load_env(f, environ={"A": "override"})
    assert env["A"] == "override" and env["B"] == "two words"
    with pytest.raises(KeyError, match="C"):
        require(env, "A", "C")



def test_select_pilot_end_to_end(tmp_path):
    ts = pd.bdate_range("2026-08-10", periods=25, tz="UTC")

    def b(vol):
        return [{"t": t.isoformat().replace("+00:00", "Z"), "o": 50, "h": 52, "l": 48,
                 "c": 50, "v": vol, "vw": 50, "n": 5} for t in ts]

    http, _ = make_http({
        "/v2/stocks/bars": [(200, {"bars": {"BIG": b(2_000_000), "SMALL": b(1_000)}})],
        "/v2/assets": [(200, [
            {"symbol": s, "name": f"{s} Inc", "exchange": "NYSE", "status": "active",
             "tradable": True, "class": "us_equity"} for s in ("BIG", "SMALL")]), (200, [])],
    })
    clock = FixedClock(NOW)
    mgr = IngestionManager(
        AlpacaBars(http, clock, RateLimiter(1000)), AlpacaReference(http),
        AlpacaNews(http, limiter=RateLimiter(1000)), ParquetBars(tmp_path),
        DatasetStore(tmp_path), clock)
    cfg = tmp_path / "u.yaml"
    cfg.write_text("min_price: 10\nmin_dollar_volume_20d: 50000000\nmin_atr_pct: 0.02\n")
    with pytest.raises(RuntimeError, match="reference"):
        mgr.select_pilot(NOW, 10, cfg)
    mgr.ingest_reference()
    checked, table = mgr.select_pilot(datetime(2026, 9, 30, tzinfo=UTC), 10, cfg)
    assert checked == 2 and list(table["symbol"]) == ["BIG"]  # BIG trades $100M a day, SMALL $50k
