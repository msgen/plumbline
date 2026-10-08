import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from signalplat.accessors.bars_parquet import ParquetBars
from signalplat.accessors.dataset_store import DatasetStore
from signalplat.accessors.experiment_store import ExperimentStore
from signalplat.contracts.types import Feed
from signalplat.managers.experiment import ExperimentManager
from signalplat.utilities.clock import FixedClock

ROOT = Path(__file__).resolve().parents[2]
NY = "America/New_York"
DAYS = list(pd.bdate_range("2026-07-01", periods=45))
BUSY = DAYS[-6:]                    # the last six days trade four times the volume


def day_path():
    """Flat opening range at 100, a breakout at 09:50, then a steady climb."""
    closes = [100.0] * 20 + [101.0 + 0.05 * i for i in range(370)]
    return closes


def minute_frame(symbol, day, volume):
    closes = day_path()
    ts = pd.date_range(f"{day:%Y-%m-%d} 09:30", periods=390, freq="min", tz=NY).tz_convert("UTC")
    c = pd.Series(closes, index=ts)
    return pd.DataFrame({
        "symbol": symbol, "timestamp": ts, "open": c.values, "high": c.values + 0.05,
        "low": c.values - 0.05, "close": c.values, "volume": volume, "vwap": c.values,
        "trade_count": 20, "available_at": ts + pd.Timedelta("1min")})


def daily_frame(symbol, day, volume):
    ts = pd.Timestamp(f"{day:%Y-%m-%d} 00:00", tz=NY).tz_convert("UTC")
    return pd.DataFrame({
        "symbol": [symbol], "timestamp": [ts], "open": 100.0, "high": 100.5, "low": 99.5,
        "close": 100.0, "volume": volume, "vwap": 100.0, "trade_count": 5,
        "available_at": [ts + pd.Timedelta("1D")]})


def build(tmp_path):
    bars, store = ParquetBars(tmp_path), DatasetStore(tmp_path)
    for day in DAYS:
        factor = 4.0 if day in BUSY else 1.0
        sip, iex = minute_frame("AAA", day, 1000 * factor), minute_frame("AAA", day, 100 * factor)
        bars.write_minute(sip, Feed.SIP)
        bars.write_minute(iex, Feed.IEX)
        bars.write_daily(daily_frame("AAA", day, 1000 * factor * 390))
    return bars, store


def config(tmp_path):
    universe = tmp_path / "universe.yaml"
    universe.write_text("min_price: 10\nmin_dollar_volume_20d: 1000\nmin_atr_pct: 0.001\n")
    e3 = tmp_path / "e3.yaml"
    e3.write_text(
        f"strategies_config: {ROOT / 'config/strategies.yaml'}\n"
        f"costs_config: {ROOT / 'config/costs.yaml'}\nuniverse_config: {universe}\n"
        "delays_seconds: [60]\nn_boot: 500\nseed: 3\n"
        "gate: {min_trades: 3, min_days: 3, min_share_years_positive: 0.5, ci_level: 0.9}\n")
    return e3


def test_e3_end_to_end_variants_and_labels(tmp_path):
    bars, store = build(tmp_path)
    mgr = ExperimentManager(bars, store, ExperimentStore(tmp_path),
                            FixedClock(datetime(2026, 9, 15, 12, 0, tzinfo=UTC)))
    res = mgr.run_e3(config(tmp_path), ["AAA"], datetime(2026, 7, 1, tzinfo=UTC),
                     datetime(2026, 9, 1, tzinfo=UTC))
    rows = {(r["strategy"], r["variant"]): r for r in res["metrics"]["summary"]}
    a_none, a_sip = rows[("A_breakout", "none")], rows[("A_breakout", "sip")]
    # the "none" variant fires on every scanned day; the volume variants only on the busy days
    assert a_none["trades"] >= 25      # 45 days minus ~15 days of universe warm-up
    assert 1 <= a_sip["trades"] <= len(BUSY)
    assert a_sip["trades"] < a_none["trades"]
    assert ("A_breakout", "iex_scaled") in rows                  # IEX at 1/10 scales back to SIP
    # the climb after the breakout reaches the 2R target, so the average is positive
    assert a_none["mean_r"] > 1.0 and a_none["target_rate"] > 0.9
    assert res["passed"] and res["verdict"]["A_breakout|none|2.0R|60s"]["status"] == "pass"
    assert res["metrics"]["trial_count"] == len(res["metrics"]["summary"])
    run_dir = tmp_path / "experiments" / res["run_id"]
    assert "rule-based setups" in (run_dir / "report.md").read_text()
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["metrics"]["costs"]["half_spread"] > 0
    # labels are kept for later experiments, with entry details
    parts = list((tmp_path / "research" / "labels").glob("part-*.parquet"))
    assert parts
    kept = pd.concat(pd.read_parquet(p) for p in parts)
    assert {"outcome", "entry_price", "r", "delay_seconds", "variant"} <= set(kept.columns)


def test_e3_with_no_universe_days_returns_a_clean_failure(tmp_path):
    bars, store = build(tmp_path)
    cfg = config(tmp_path)
    (tmp_path / "universe.yaml").write_text(
        "min_price: 500\nmin_dollar_volume_20d: 1000\nmin_atr_pct: 0.001\n")
    mgr = ExperimentManager(bars, store, ExperimentStore(tmp_path),
                            FixedClock(datetime(2026, 9, 15, 12, 0, tzinfo=UTC)))
    res = mgr.run_e3(cfg, ["AAA"], datetime(2026, 7, 1, tzinfo=UTC),
                     datetime(2026, 9, 1, tzinfo=UTC))
    assert not res["passed"] and res["metrics"]["signals"] == 0
