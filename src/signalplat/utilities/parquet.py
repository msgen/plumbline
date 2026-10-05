"""Immutable Parquet parts: files are only ever added, never edited. DuckDB reads them back."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from signalplat.utilities.ids import hash_bytes


def write_part(df: pd.DataFrame, directory: Path) -> Path:
    """Write df as a new content-addressed part file; identical content is a no-op."""
    directory.mkdir(parents=True, exist_ok=True)
    body = df.to_parquet(index=False)
    path = directory / f"part-{hash_bytes(body)[:16]}.parquet"
    if not path.exists():
        path.write_bytes(body)
    return path


def read_parts(
    directory: Path, where: str = "true", params: list | None = None
) -> pd.DataFrame | None:
    """Read every part under directory (recursively), de-duplicated. None if there are none."""
    if not directory.exists() or not any(directory.rglob("part-*.parquet")):
        return None
    glob = str(directory / "**" / "part-*.parquet")
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT DISTINCT * FROM read_parquet(?, union_by_name=true) WHERE {where}",  # noqa: S608
            [glob, *(params or [])],
        ).df()
    finally:
        con.close()


def fingerprint(directory: Path) -> str:
    """Content hash of a dataset. Part names are content hashes, so names identify the data."""
    names = sorted(str(p.relative_to(directory)) for p in directory.rglob("part-*.parquet"))
    return hash_bytes("\n".join(names).encode())
