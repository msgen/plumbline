"""Command-line client: one subcommand per manager."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from signalplat.contracts.types import Feed
from signalplat.managers.experiment import ExperimentManager
from signalplat.managers.ingestion import IngestionManager
from signalplat.utilities.http import HttpError


def _day(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)


def _symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols_file:
        path = Path(args.symbols_file)
        if not path.exists():
            raise SystemExit(
                f"symbols file not found: {path} (try config/pilot_small.txt, "
                "or pass --symbols AAPL,MSFT)")
        text = path.read_text(encoding="utf-8")
    else:
        text = args.symbols.replace(",", "\n")
    return sorted({s.strip().upper() for s in text.splitlines() if s.strip()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signalplat")
    sub = parser.add_subparsers(dest="command", required=True)
    ing = sub.add_parser("ingest", help="download and store raw data")
    ing.add_argument("--symbols", default="", help="comma-separated symbols")
    ing.add_argument("--symbols-file", help="file with one symbol per line")
    ing.add_argument("--start", type=_day, required=True, help="YYYY-MM-DD (UTC)")
    ing.add_argument("--end", type=_day, required=True, help="YYYY-MM-DD (UTC, exclusive)")
    steps = ["reference", "daily", "splits", "minute", "news", "filings"]
    ing.add_argument("--only", nargs="+", choices=steps, default=steps)
    ing.add_argument("--delisted-sample", type=int, default=0, metavar="N",
                     help="also fetch daily bars for N sampled inactive symbols (for E0)")
    ing.add_argument("--seed", type=int, default=7)
    ing.add_argument("--delisted-start", type=_day, default=_day("2016-01-01"),
                     help="history start for the delisted sample (default 2016-01-01)")
    ing.add_argument("--feeds", nargs="+", choices=[f.value for f in Feed], default=["sip", "iex"])
    pil = sub.add_parser("pilot", help="pick the most liquid stocks that pass the universe filters")
    pil.add_argument("--as-of", type=_day, required=True, help="YYYY-MM-DD; uses data before it")
    pil.add_argument("--n", type=int, default=200)
    pil.add_argument("--config", default="config/universe.yaml")
    pil.add_argument("--out", default="config/pilot.txt")
    ins = sub.add_parser("inspect", help="show stored bars facts for one symbol and day")
    ins.add_argument("symbol")
    ins.add_argument("day", type=_day, help="YYYY-MM-DD (New York trading day)")
    ins.add_argument("--feed", choices=[f.value for f in Feed], default="sip")
    aud = sub.add_parser("audit", help="run the E0 data audit on stored data")
    aud.add_argument("--symbols", default="", help="comma-separated symbols")
    aud.add_argument("--symbols-file", help="file with one symbol per line")
    aud.add_argument("--start", type=_day, required=True)
    aud.add_argument("--end", type=_day, required=True)
    aud.add_argument("--config", default="experiments/E00_data_audit.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except HttpError as e:
        print(f"error: {e}")
        return 1


def _run(args: argparse.Namespace) -> int:
    if args.command == "ingest":
        mgr = IngestionManager.from_env()
        symbols = _symbols(args)
        if "reference" in args.only:
            print(f"assets: {mgr.ingest_reference()}")
        if symbols and "daily" in args.only:
            print(f"daily bars: {mgr.ingest_daily(symbols, args.start, args.end)}")
        if symbols and "splits" in args.only:
            print(f"splits: {mgr.ingest_splits(symbols, args.start, args.end)}")
        if symbols and "minute" in args.only:
            for feed in args.feeds:
                n = mgr.ingest_minute(symbols, args.start, args.end, Feed(feed))
                print(f"minute bars ({feed}): {n}")
        if symbols and "news" in args.only:
            print(f"news: {mgr.ingest_news(symbols, args.start, args.end)}")
        if symbols and "filings" in args.only:
            print(f"filings: {mgr.ingest_filings(symbols, args.start, args.end)}")
        if args.delisted_sample:
            sample = mgr.ingest_delisted_sample(
                args.delisted_sample, args.seed, args.delisted_start, args.end)
            print(f"delisted sample daily bars fetched for {len(sample)} symbols")
        if mgr.unmapped_filing_symbols:
            print(f"no EDGAR CIK (filings missing) for {len(mgr.unmapped_filing_symbols)} "
                  f"symbols: {', '.join(sorted(mgr.unmapped_filing_symbols)[:30])}")
        if mgr.skipped_symbols:
            print(f"vendor rejected {len(mgr.skipped_symbols)} symbols: "
                  f"{', '.join(sorted(mgr.skipped_symbols))}")
    elif args.command == "pilot":
        checked, table = IngestionManager.from_env().select_pilot(
            args.as_of, args.n, args.config)
        Path(args.out).write_text("\n".join(table["symbol"]) + "\n", encoding="utf-8")
        print(f"{checked} eligible symbols checked, {len(table)} written to {args.out}")
        print(table.head(10).to_string(index=False))
    elif args.command == "inspect":
        info = ExperimentManager.from_env().inspect_day(
            args.symbol.upper(), args.day, Feed(args.feed))
        for key, value in info.items():
            print(f"{key}: {value}")
    elif args.command == "audit":
        result = ExperimentManager.from_env().run_e0(
            args.config, _symbols(args), args.start, args.end)
        print((Path(result["directory"]) / "report.md").read_text(encoding="utf-8"))
        print(f"saved to {result['directory']}")
        return 0 if result["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
