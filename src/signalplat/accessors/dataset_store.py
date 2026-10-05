"""Local Parquet storage for reference data, news and the ingestion manifest."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from signalplat.contracts.types import NewsItem
from signalplat.utilities.parquet import read_parts, write_part


class DatasetStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def write_assets(self, df: pd.DataFrame) -> None:
        write_part(df, self._root / "reference" / "assets")

    def read_assets(self) -> pd.DataFrame | None:
        return read_parts(self._root / "reference" / "assets")

    def write_news(self, items: Sequence[NewsItem]) -> int:
        if not items:
            return 0
        df = pd.DataFrame([
            {"id": i.id, "created_at": i.created_at, "available_at": i.available_at,
             "headline": i.headline, "summary": i.summary, "symbols": list(i.symbols)}
            for i in items
        ])
        for month, g in df.groupby(df["created_at"].dt.strftime("%Y-%m")):
            write_part(g, self._root / "news" / f"month={month}")
        return len(df)

    def write_filings(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        for year, g in df.groupby(df["accepted_at"].dt.year):
            write_part(g, self._root / "edgar" / f"year={year}")
        return len(df)

    def read_filings(self) -> pd.DataFrame | None:
        return read_parts(self._root / "edgar")

    # The manifest lets an interrupted backfill resume without refetching finished chunks.
    def is_done(self, key: str) -> bool:
        path = self._root / "_manifest.jsonl"
        if not path.exists():
            return False
        return any(json.loads(line)["key"] == key for line in path.read_text().splitlines() if line)

    def mark_done(self, key: str, rows: int) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        with open(self._root / "_manifest.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "rows": rows}) + "\n")
