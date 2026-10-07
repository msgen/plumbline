"""Experiment manager: the sequence of a run (load data, compute, gate, report, store)."""
from __future__ import annotations

from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

from signalplat.accessors.bars_parquet import ParquetBars
from signalplat.accessors.dataset_store import DatasetStore
from signalplat.accessors.experiment_store import ExperimentStore
from signalplat.contracts.types import Feed
from signalplat.engines import quality
from signalplat.engines.adjust import adjust_for_splits
from signalplat.engines.universe import membership
from signalplat.utilities.clock import Clock, SystemClock
from signalplat.utilities.config import load_config
from signalplat.utilities.env import load_env
from signalplat.utilities.ids import hash_config
from signalplat.utilities.logging import get_logger

log = get_logger("experiment")


class ExperimentManager:
    def __init__(
        self, bars: ParquetBars, store: DatasetStore, runs: ExperimentStore, clock: Clock
    ) -> None:
        self._bars, self._store, self._runs, self._clock = bars, store, runs, clock

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ExperimentManager:
        env = env if env is not None else load_env()
        root = Path(env.get("DATA_DIR", "./data"))
        return cls(ParquetBars(root), DatasetStore(root), ExperimentStore(root), SystemClock())

    def run_e0(
        self, config_path: str | Path, symbols: list[str], start: datetime, end: datetime
    ) -> dict[str, Any]:
        cfg = load_config(config_path)
        gate = cfg["gate"]
        log.info("summarising minute bars (SIP)")
        sip = self._bars.minute_summary(symbols, start, end, Feed.SIP)
        log.info("summarising minute bars (IEX)")
        iex = self._bars.minute_summary(symbols, start, end, Feed.IEX)
        daily = self._bars.daily_bars(symbols, start, end)

        assets = self._store.read_assets()
        sample = (
            quality.sample_delisted(assets, cfg["delisted_sample"], cfg["seed"])
            if assets is not None else []
        )
        delisted_start = datetime.combine(cfg["delisted_start"], time(), tzinfo=UTC)
        delisted_daily = self._bars.daily_bars(sample, min(start, delisted_start), end)

        # Gates apply to the point-in-time universe, so judge each day on earlier bars only.
        daily_adj = adjust_for_splits(
            daily.assign(raw_close=daily["close"]), self._store.read_splits(), end)
        members = membership(daily_adj, load_config(cfg["universe_config"]))
        all_jumps = quality.price_jumps(daily_adj)
        listed = quality.price_jumps(daily_adj, members=members)
        reviewed = {(r["symbol"], str(r["day"])) for r in cfg.get("reviewed_jumps") or []}
        jumps = [j for j in listed if (j[0], j[1]) not in reviewed]
        n_reviewed = len(listed) - len(jumps)

        metrics = {
            "delisted": quality.delisted_coverage(sample, delisted_daily),
            "missing_minutes": quality.missing_regular_minutes(sip, daily, members),
            "missing_minutes_iex": quality.missing_regular_minutes(iex, daily, members),
            "volume": quality.volume_consistency(sip, daily, gate["daily_vs_minute_volume_tol"]),
            "duplicates": quality.duplicate_minute_rows(sip) + quality.duplicate_count(daily),
            "outside_hours": quality.outside_extended_hours(sip),
            "boundary_bars": quality.boundary_bars(sip),
            "price_jumps_reviewed": n_reviewed,
            "outside_hours_examples": self._bars.outside_hours_examples(
                symbols, start, end, Feed.SIP),
            "price_jumps_outside_universe": len(all_jumps) - len(listed),
            # judged after known splits are applied, on universe days, minus reviewed ones
            "price_jumps": jumps,
            "filings_coverage": quality.filings_coverage(
                symbols, sample, self._store.read_filings()),
            "news_coverage": quality.news_coverage(
                self._store.read_news(), symbols, (end - start).days),
        }
        verdict = quality.evaluate_e0(metrics, gate)
        failed = [k for k, v in verdict.items() if v["status"] == "fail"]
        run_id = f"E0-{self._clock.now():%Y%m%dT%H%M%SZ}"
        record = {
            "run_id": run_id, "experiment": "E0", "config": cfg,
            "config_hash": hash_config(cfg), "code_commit": self._runs.code_commit(),
            "dataset_hashes": self._runs.dataset_hash(
                "bars_raw_1d", "bars_raw_1m", "reference", "news", "corporate_actions", "edgar"),
            "window": [start, end], "symbols": symbols, "delisted_sample": sample,
            "metrics": metrics, "verdict": verdict, "passed": not failed,
        }
        directory = self._runs.write_run(run_id, record, render_e0(record))
        return {**record, "directory": str(directory)}


def render_e0(r: dict[str, Any]) -> str:
    lines = [f"# {r['run_id']}: data audit", "",
             f"Result: **{'PASS' if r['passed'] else 'FAIL'}** "
             "(gates are fixed; a failure means redesign or fetch better data, not a looser gate)",
             "", "| Check | Value | Gate | Status |", "|---|---|---|---|"]
    for name, v in r["verdict"].items():
        val = "n/a" if v["value"] is None else (
            f"{v['value']:.4f}" if isinstance(v["value"], float) else v["value"])
        lines.append(f"| {name} | {val} | {v['op']} {v['threshold']} | {v['status']} |")
    m = r["metrics"]
    lines += ["", f"Delisted sample: {m['delisted']['present']}/{m['delisted']['sampled']} "
              f"present. Missing: {', '.join(m['delisted']['missing'][:20]) or 'none'}",
              "", "Catalyst data coverage (informational; no gate is set yet):",
              f"- filings, universe: {m['filings_coverage']['universe']}",
              f"- filings, delisted sample: {m['filings_coverage']['delisted_sample']} "
              "(a big gap means the dilution veto would be survivorship-biased)",
              f"- news: {m['news_coverage']}",
              "", "Delisted names absent from Alpaca's asset list cannot be measured here; "
              "compare against an external list if this matters.",
              "IEX missing-minute fraction (informational): "
              f"{m['missing_minutes_iex']['fraction']}",
              f"Volume convention matched: {m['volume'].get('convention')}",
              "", f"Missing-minute scope for the gate: {m['missing_minutes']['scope']} "
              f"({m['missing_minutes']['symbol_days']} symbol-days). "
              f"All days, including days a stock was not in the universe: "
              f"{m['missing_minutes']['fraction_all']}",
              f"Bars stamped exactly 20:00 (session boundary, not gated): {m['boundary_bars']}",
              "Bars before 04:00 or after 20:00 New York (examples): "
              f"{m['outside_hours_examples']}",
              "Missing-minute fraction by regular-session trades per minute "
              "(universe days):",
              *[f"- {k}: {v}" for k, v in m["missing_minutes"]["by_trades_per_minute"].items()],
              f"Price jumps outside the universe (not listed): {m['price_jumps_outside_universe']}",
              "", "Worst symbol-days for missing minutes (symbol, day, minutes):",
              *[f"- {w}" for w in m["missing_minutes"]["worst"]],
              "", f"Price jumps over 40% still needing review ({len(m['price_jumps'])}; "
              f"{m['price_jumps_reviewed']} already marked reviewed in the config):",
              *[f"- {j}" for j in m["price_jumps"][:50]]]
    return "\n".join(lines) + "\n"
