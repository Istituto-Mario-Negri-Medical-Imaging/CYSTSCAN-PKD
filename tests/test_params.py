"""Published parameters live in one file; the GA is reproducible with a seed."""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_VALUES = [4.27269938837114, 18.2632731580751, 42144.3492985436,
                14.9647743085524, 0.111299008136904, 0.918315428393029]
SKIP_DIRS = {".venv", ".git", "data", "results", "models", "tests",
             "build", "dist", "__pycache__"}
TEXT_SUFFIXES = {".py", ".m", ".R", ".r", ".md", ".json", ".toml", ".cff", ".txt"}


def test_values_match_paper():
    from cystscan.params import load_published_params

    np.testing.assert_array_equal(load_published_params(), PAPER_VALUES)


def test_no_hardcoded_copies():
    config = REPO_ROOT / "cystscan" / "data" / "published_params.json"
    offenders = []
    for path in REPO_ROOT.rglob("*"):
        rel = path.relative_to(REPO_ROOT)
        if (not path.is_file() or path.suffix not in TEXT_SUFFIXES
                or SKIP_DIRS.intersection(rel.parts) or path == config):
            continue
        if "4.27269938837114" in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(str(rel))
    assert offenders == [], f"published values hardcoded in: {offenders}"


def _load_run_optimization():
    spec = importlib.util.spec_from_file_location(
        "run_optimization", REPO_ROOT / "python" / "run_optimization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ga_seed_reproducible():
    ga = _load_run_optimization()
    ga.seed_everything(42)
    first = [ga.random_individual() for _ in range(3)]
    ga.seed_everything(42)
    second = [ga.random_individual() for _ in range(3)]
    assert first == second
    ga.seed_everything(43)
    assert [ga.random_individual() for _ in range(3)] != first
    random.seed()  # leave global state unseeded for other tests
