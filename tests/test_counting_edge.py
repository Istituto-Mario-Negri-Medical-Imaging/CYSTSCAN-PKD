"""Edge cases of the cyst counting, pinned on small synthetic masks."""
from __future__ import annotations

import nibabel as nib
import numpy as np

from cystscan.counting import PUBLISHED_PARAMS, detect_cyst_peaks


def _two_cyst_mask() -> np.ndarray:
    """40^3 kidney (label 1) with two separate spherical cysts (label 2)."""
    mask = np.ones((40, 40, 40), dtype=np.int16)
    zz, yy, xx = np.mgrid[:40, :40, :40]
    for center in ((12, 12, 12), (28, 28, 28)):
        r2 = sum((g - c) ** 2 for g, c in zip((zz, yy, xx), center))
        mask[r2 <= 5 ** 2] = 2
    return mask


def _save(tmp_path, name, array, spacing=0.0294):
    path = tmp_path / name
    nib.save(nib.Nifti1Image(array, np.diag([spacing] * 3 + [1.0])), path)
    return str(path)


def test_empty_mask_returns_zero(tmp_path):
    mask = np.ones((20, 20, 20), dtype=np.int16)
    mask[:5] = 0
    assert detect_cyst_peaks(_save(tmp_path, "empty.nii.gz", mask), PUBLISHED_PARAMS) == 0


def test_4d_mask_equals_3d(tmp_path):
    mask = _two_cyst_mask()
    count_3d = detect_cyst_peaks(_save(tmp_path, "m3d.nii.gz", mask), PUBLISHED_PARAMS)
    count_4d = detect_cyst_peaks(_save(tmp_path, "m4d.nii.gz", mask[..., None]),
                                 PUBLISHED_PARAMS)
    assert count_3d > 0
    assert count_4d == count_3d


def test_spacing_ignored(tmp_path):
    """The counting works in voxels: the same array at another spacing gives the same count."""
    mask = _two_cyst_mask()
    fine = detect_cyst_peaks(_save(tmp_path, "s1.nii.gz", mask, 0.0195), PUBLISHED_PARAMS)
    coarse = detect_cyst_peaks(_save(tmp_path, "s2.nii.gz", mask, 0.0294), PUBLISHED_PARAMS)
    assert fine == coarse
