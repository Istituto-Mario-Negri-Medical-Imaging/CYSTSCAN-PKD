"""Counting on the synthetic demo fixture (tests/data, made by tools/make_demo_fixtures.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cystscan.counting import PUBLISHED_PARAMS, detect_cyst_peaks

DATA = Path(__file__).resolve().parent / "data"
EXPECTED = json.loads((DATA / "expected_counts.json").read_text())


def test_synthetic_count_exact():
    # 12 separate spheres, one cyst each
    assert detect_cyst_peaks(str(DATA / "synthetic_mask.nii.gz"), PUBLISHED_PARAMS) == 12


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_expected_counts(name):
    assert detect_cyst_peaks(str(DATA / f"{name}.nii.gz"), PUBLISHED_PARAMS) == EXPECTED[name]


def test_fixture_size():
    files = list(DATA.glob("*.nii.gz"))
    assert files
    for f in files:
        assert f.stat().st_size < 1_000_000, f.name
