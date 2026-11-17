"""Python counting reproduces the paper's MATLAB counts on the Evaluation set.

Needs the Zenodo data in data/ (skipped otherwise, e.g. in CI).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cystscan.counting import PUBLISHED_PARAMS, detect_cyst_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
MASK_DIR = REPO_ROOT / "data" / "2_Cyst_counting_pipeline" / "Evaluation set" / "Cyst masks"

# detectCystPeaks.m with the published parameters (paper; python/README.md, MATLAB column)
MATLAB_COUNTS = {
    "KRATS_005": 89, "KRATS_014": 88, "KRATS_018": 119, "KRATS_025": 101,
    "KRATS_038": 83, "KRATS_066": 77, "XRATS_018": 243, "XRATS_033": 188,
    "XRATS_046": 82, "XRATS_050": 137,
}


@pytest.mark.skipif(not MASK_DIR.is_dir(), reason="Evaluation set not in data/")
@pytest.mark.parametrize("case", sorted(MATLAB_COUNTS))
def test_counts_match_matlab(case):
    count = detect_cyst_peaks(str(MASK_DIR / f"{case}.nii.gz"), PUBLISHED_PARAMS)
    assert count == MATLAB_COUNTS[case]
