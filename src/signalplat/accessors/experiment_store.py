"""Experiment runs on disk: config, code commit, dataset hashes, metrics, report."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from signalplat.utilities.parquet import fingerprint


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def dataset_hash(self, *names: str) -> dict[str, str | None]:
        """Fingerprint of each named dataset directory under the data root (None if absent)."""
        return {
            n: fingerprint(self._root / n) if (self._root / n).exists() else None for n in names
        }

    @staticmethod
    def code_commit() -> str:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    def write_run(self, run_id: str, record: dict[str, Any], report_md: str) -> Path:
        directory = self._root / "experiments" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8")
        (directory / "report.md").write_text(report_md, encoding="utf-8")
        return directory
