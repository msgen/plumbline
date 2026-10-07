"""Experiment manager: the sequence of a run (load data, compute, gate, report, store)."""
from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

from signalplat.accessors.bars_parquet import ParquetBars
from signalplat.accessors.dataset_store import DatasetStore
from signalplat.accessors.experiment_store import ExperimentStore
from signalplat.contracts.types import Feed
from signalplat.engines import feeds, quality
from signalplat.engines.adjust import adjust_for_splits
from signalplat.engines.universe import membership
from signalplat.utilities.clock import NY, Clock, SystemClock, regular_close_minute
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

    def inspect_day(self, symbol: str, day: datetime, feed: Feed = Feed.SIP) -> dict[str, Any]:
        """Facts about one symbol-day of stored bars, to judge a suspicious audit row by eye."""
        lo, hi = day - timedelta(hours=6), day + timedelta(hours=30)  # covers the NY day
        minute = self._bars.minute_bars([symbol], lo, hi, feed)
        daily = self._bars.daily_bars([symbol], day - timedelta(days=1), day + timedelta(days=2))
        if minute.empty:
            return {"symbol": symbol, "day": str(day.date()), "minute_bars": 0,
                    "daily_bars": daily[["timestamp", "close", "volume"]].to_dict("records")}
        local = minute["timestamp"].dt.tz_convert(NY)
        minute = minute[local.dt.date == day.date()]
        mod = (local.dt.hour * 60 + local.dt.minute)[minute.index]
        reg = (mod >= 570) & (mod < regular_close_minute(day.date()))
        return {
            "symbol": symbol, "day": str(day.date()), "feed": feed.value,
            "minute_bars": len(minute), "regular_minute_bars": int(reg.sum()),
            "first": f"{mod.min() // 60:02d}:{mod.min() % 60:02d}",
            "last": f"{mod.max() // 60:02d}:{mod.max() % 60:02d}",
            "minute_volume_all": float(minute["volume"].sum()),
            "minute_volume_regular": float(minute.loc[reg, "volume"].sum()),
            "daily_volume": [float(v) for v in daily["volume"]],
            "trades_regular": float(minute.loc[reg, "trade_count"].sum()),
            "longest_gaps_regular": quality.minute_gaps(mod[reg].tolist()),
        }

    def _universe_days(
        self, symbols: list[str], start: datetime, end: datetime, ucfg: dict[str, Any]
    ):
        """Raw daily bars, their split-adjusted copy, and point-in-time universe membership."""
        daily = self._bars.daily_bars(symbols, start, end)
        daily_adj = adjust_for_splits(
            daily.assign(raw_close=daily["close"]), self._store.read_splits(), end)
        return daily, daily_adj, membership(daily_adj, ucfg)

    def _save(
        self, experiment: str, cfg: dict[str, Any], symbols: list[str], start: datetime,
        end: datetime, metrics: dict[str, Any], verdict: dict[str, Any], render, extra=None,
    ) -> dict[str, Any]:
        failed = [k for k, v in verdict.items() if v["status"] == "fail"]
        run_id = f"{experiment}-{self._clock.now():%Y%m%dT%H%M%SZ}"
        record = {
            "run_id": run_id, "experiment": experiment, "config": cfg,
            "config_hash": hash_config(cfg), "code_commit": self._runs.code_commit(),
            "dataset_hashes": self._runs.dataset_hash(
                "bars_raw_1d", "bars_raw_1m", "reference", "news", "corporate_actions", "edgar"),
            "window": [start, end], "symbols": symbols, **(extra or {}),
            "metrics": metrics, "verdict": verdict, "passed": not failed,
        }
        directory = self._runs.write_run(run_id, record, render(record))
        return {**record, "directory": str(directory)}

    def run_e1(
        self, config_path: str | Path, symbols: list[str], start: datetime, end: datetime
    ) -> dict[str, Any]:
        """E1: can scaled IEX volume replace SIP volume for RVOL and VWAP near the open?"""
        cfg = load_config(config_path)
        ucfg = load_config(cfg["universe_config"])
        minutes = [int(t[:2]) * 60 + int(t[3:]) for t in cfg["times"]]
        _, _, members = self._universe_days(symbols, start, end, ucfg)
        log.info("summarising opening volume (SIP)")
        sip = self._bars.opening_volume(symbols, start, end, Feed.SIP, minutes)
        log.info("summarising opening volume (IEX)")
        iex = self._bars.opening_volume(symbols, start, end, Feed.IEX, minutes)
        study = feeds.rvol_study(
            sip, iex, members, minutes, cfg["trailing_days"], cfg["min_history_days"],
            cfg["rvol_flag"])
        error_label = feeds.label(int(cfg["error_time"][:2]) * 60 + int(cfg["error_time"][3:]))
        flag_label = feeds.label(int(cfg["flag_time"][:2]) * 60 + int(cfg["flag_time"][3:]))
        verdict = feeds.evaluate_e1(study, cfg["gate"], error_label, flag_label)
        return self._save("E1", cfg, symbols, start, end, study, verdict, render_e1)

    def run_e0(
        self, config_path: str | Path, symbols: list[str], start: datetime, end: datetime
    ) -> dict[str, Any]:
        cfg = load_config(config_path)
        gate = cfg["gate"]
        log.info("summarising minute bars (SIP)")
        sip = self._bars.minute_summary(symbols, start, end, Feed.SIP)
        log.info("summarising minute bars (IEX)")
        iex = self._bars.minute_summary(symbols, start, end, Feed.IEX)

        assets = self._store.read_assets()
        sample = (
            quality.sample_delisted(assets, cfg["delisted_sample"], cfg["seed"])
            if assets is not None else []
        )
        delisted_start = datetime.combine(cfg["delisted_start"], time(), tzinfo=UTC)
        delisted_daily = self._bars.daily_bars(sample, min(start, delisted_start), end)

        # Gates apply to the point-in-time universe, so judge each day on earlier bars only.
        ucfg = load_config(cfg["universe_config"])
        daily, daily_adj, members = self._universe_days(symbols, start, end, ucfg)
        # Minute bars skip odd-lot trades, so they are complete only for cheaper stocks. The
        # completeness gate covers universe days below this price; the rest is reported.
        cap = ucfg.get("complete_bars_max_price")
        gated = members if cap is None else members.assign(
            member=members["member"] & (members["price"] < cap))
        all_jumps = quality.price_jumps(daily_adj)
        listed = quality.price_jumps(daily_adj, members=members)
        reviewed = {(r["symbol"], str(r["day"])) for r in cfg.get("reviewed_jumps") or []}
        jumps = [j for j in listed if (j[0], j[1]) not in reviewed]
        n_reviewed = len(listed) - len(jumps)

        metrics = {
            "delisted": quality.delisted_coverage(sample, delisted_daily),
            "missing_minutes": quality.missing_regular_minutes(sip, daily, gated),
            "missing_minutes_full": quality.missing_regular_minutes(sip, daily, members),
            "missing_minutes_iex": quality.missing_regular_minutes(iex, daily, gated),
            "complete_bars_max_price": cap,
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
        return self._save("E0", cfg, symbols, start, end, metrics, verdict, render_e0,
                          {"delisted_sample": sample})


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
    cap = m.get("complete_bars_max_price")
    lines += ["", "Missing-minute gate covers universe days "
              + (f"priced below ${cap:g}." if cap else "at any price."), "",
              f"Delisted sample: {m['delisted']['present']}/{m['delisted']['sampled']} "
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
              *[f"- {k}: {v}"
                for k, v in m["missing_minutes_full"]["by_trades_per_minute"].items()],
              f"Price jumps outside the universe (not listed): {m['price_jumps_outside_universe']}",
              "", "Whole universe, any price (informational): missing fraction "
              f"{m['missing_minutes_full']['fraction']}",
              "Missing-minute fraction by raw daily close (universe days):",
              *[f"- {k}: {v}" for k, v in m["missing_minutes_full"]["by_price"].items()],
              "Per-symbol missing fraction quantiles: "
              f"{m['missing_minutes_full']['per_symbol_fraction_quantiles']}",
              "Symbols with the highest missing fraction (symbol, fraction, median price): "
              f"{m['missing_minutes_full']['worst_symbols']}",
              "", "Days with over 10% of minutes missing, volume check (do the empty minutes carry "
              f"volume?): {m['missing_minutes_full']['gappy_days_volume']}",
              "", "Worst symbol-days for missing minutes (symbol, day, minutes missing, "
              "volume gap vs daily bar):",
              *[f"- {w}" for w in m["missing_minutes"]["worst"]],
              "", f"Price jumps over 40% still needing review ({len(m['price_jumps'])}; "
              f"{m['price_jumps_reviewed']} already marked reviewed in the config):",
              *[f"- {j}" for j in m["price_jumps"][:50]]]
    return "\n".join(lines) + "\n"


def render_e1(r: dict[str, Any]) -> str:
    cfg = r["config"]
    lines = [f"# {r['run_id']}: IEX versus SIP near the open", "",
             f"Result: **{'PASS' if r['passed'] else 'FAIL'}** "
             "(a fail means: pay for the full feed, start signals later, or use price-only "
             "features at the open)", "",
             "| Check | Value | Gate | Status |", "|---|---|---|---|"]
    for name, v in r["verdict"].items():
        val = "n/a" if v["value"] is None else f"{v['value']:.4f}"
        lines.append(f"| {name} | {val} | {v['op']} {v['threshold']} | {v['status']} |")
    lines += ["", "Universe days only. RVOL = volume since 09:30 over the mean of the previous "
              f"{cfg['trailing_days']} days at the same time. IEX is scaled by its trailing "
              "SIP/IEX ratio. Flag agreement counts the days either feed flags "
              f"(RVOL >= {cfg['rvol_flag']}), not the many days neither does.", ""]
    for lab, v in r["metrics"].items():
        if not v.get("n"):
            lines += [f"## {lab[:2]}:{lab[2:]}: no comparable days", ""]
            continue
        f = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
        lines += [
            f"## {lab[:2]}:{lab[2:]} ({v['n']} symbol-days)",
            f"- RVOL error: median {f(v['median_rvol_error'])}, p90 {f(v['p90_rvol_error'])}",
            f"- flag agreement, all days {f(v['flag_agreement_all'])}; on flagged days "
            f"{f(v['flag_agreement_flagged'])} ({v['flagged_cases']} cases); "
            f"precision {f(v['flag_precision'])}, recall {f(v['flag_recall'])}",
            f"- VWAP difference: median {v['median_vwap_bps']:.1f} bps, "
            f"p90 {v['p90_vwap_bps']:.1f} bps",
            f"- median RVOL error by price: {v['median_rvol_error_by_price']}", ""]
    return "\n".join(lines) + "\n"
