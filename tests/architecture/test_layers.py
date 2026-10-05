import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINT = shutil.which("lint-imports") or "lint-imports"


def _lint(cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [LINT],
        cwd=cwd, capture_output=True, text=True,
        env={"PYTHONPATH": str(cwd / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )


def test_layering_holds():
    r = _lint(ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 broken" in r.stdout


def test_illegal_import_fails_the_build(tmp_path):
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    bad = tmp_path / "src/signalplat/engines/universe.py"
    bad.write_text("from signalplat.accessors import bars_alpaca  # noqa\n")
    r = subprocess.run(
        [LINT],
        cwd=tmp_path, capture_output=True, text=True,
        env={"PYTHONPATH": str(tmp_path / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert r.returncode != 0
    assert "BROKEN" in r.stdout
