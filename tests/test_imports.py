"""The package imports cleanly and no script patches sys.path."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".git", "build", "dist", "__pycache__"}
FORBIDDEN = "sys.path" + ".insert"  # split so this file does not match itself


def test_package_imports():
    from cystscan.counting import detect_cyst_peaks
    from cystscan.evaluation import run_evaluation_core
    from cystscan.segmentation.inference import run_inference

    assert callable(detect_cyst_peaks)
    assert callable(run_evaluation_core)
    assert callable(run_inference)


def test_no_sys_path_hacks():
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        if SKIP_DIRS.intersection(path.relative_to(REPO_ROOT).parts):
            continue
        if FORBIDDEN in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"sys.path patched in: {offenders}"
