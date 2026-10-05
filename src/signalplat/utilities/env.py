"""Minimal .env loader. Secrets stay in a git-ignored file, never in code."""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env", environ: dict[str, str] | None = None) -> dict[str, str]:
    """Read KEY=VALUE lines; real environment variables win over the file."""
    values: dict[str, str] = {}
    p = Path(path)
    if p.exists():
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            values[key.strip()] = val
    values.update(os.environ if environ is None else environ)
    return values


def require(env: dict[str, str], *keys: str) -> None:
    missing = [k for k in keys if not env.get(k)]
    if missing:
        raise KeyError(f"missing required settings in .env: {', '.join(missing)}")
