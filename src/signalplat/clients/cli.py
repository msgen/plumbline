"""Command-line client: one subcommand per manager."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from signalplat.contracts.types import Feed
from signalplat.managers.ingestion import IngestionManager


def _day(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)


def _symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols_file:
        text = Path(args.symbols_file).read_text(encoding="utf-8")
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
    ing.add_argument("--only", nargs="+", choices=["reference", "daily", "minute", "news"],
                     default=["reference", "daily", "minute", "news"])
    ing.add_argument("--feeds", nargs="+", choices=[f.value for f in Feed], default=["sip", "iex"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        mgr = IngestionManager.from_env()
        symbols = _symbols(args)
        if "reference" in args.only:
            print(f"assets: {mgr.ingest_reference()}")
        if symbols and "daily" in args.only:
            print(f"daily bars: {mgr.ingest_daily(symbols, args.start, args.end)}")
        if symbols and "minute" in args.only:
            for feed in args.feeds:
                n = mgr.ingest_minute(symbols, args.start, args.end, Feed(feed))
                print(f"minute bars ({feed}): {n}")
        if symbols and "news" in args.only:
            print(f"news: {mgr.ingest_news(symbols, args.start, args.end)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
