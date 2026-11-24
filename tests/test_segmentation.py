"""Segmentation: input validation (no GPU) and a local end-to-end GPU check.

The GPU test runs only with an NVIDIA GPU, the [segmentation] extra and two
environment variables: CYSTSCAN_TEST_SCAN (a scan, *_0000.nii.gz) and
CYSTSCAN_TEST_REFERENCE (its reference mask, labels 0/1/2). It passes on the
cyst Dice; the cyst counts on the predicted and the reference mask are printed
for the record, not asserted: the published counts come from manually
corrected masks, and the peak-based count is sensitive to small boundary
changes. Model weights are read from CYSTSCAN_MODELS_DIR (default ./models),
and downloaded there if absent. Set CYSTSCAN_TEST_OUT to keep the predicted
mask in that folder.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from cystscan.segmentation.inference import run_inference

CYST_LABEL = 2


def test_unknown_model_raises(tmp_path):
    scan = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), np.eye(4)), scan)
    with pytest.raises(ValueError, match="Unknown model"):
        run_inference(scan, "n7", tmp_path / "out", tmp_path / "models")


def test_missing_scan_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        run_inference(tmp_path / "absent.nii.gz", "n5", tmp_path / "out",
                      tmp_path / "models")


def test_non_3d_scan_raises(tmp_path):
    scan = tmp_path / "scan4d.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 2), np.int16), np.eye(4)), scan)
    with pytest.raises(ValueError, match="3-D"):
        run_inference(scan, "n5", tmp_path / "out", tmp_path / "models")


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else 2.0 * inter / total


@pytest.mark.gpu
def test_segmentation_gpu(tmp_path):
    scan = os.environ.get("CYSTSCAN_TEST_SCAN")
    reference = os.environ.get("CYSTSCAN_TEST_REFERENCE")
    if not (scan and reference):
        pytest.skip("CYSTSCAN_TEST_SCAN and CYSTSCAN_TEST_REFERENCE not set")
    models_dir = Path(os.environ.get("CYSTSCAN_MODELS_DIR", "models"))
    out_dir = Path(os.environ.get("CYSTSCAN_TEST_OUT", tmp_path / "out"))

    from cystscan.counting import PUBLISHED_PARAMS, detect_cyst_peaks

    start = time.monotonic()
    mask_path = run_inference(Path(scan), "n5", out_dir, models_dir)
    minutes = (time.monotonic() - start) / 60

    pred = np.asarray(nib.load(mask_path).dataobj) == CYST_LABEL
    ref = np.asarray(nib.load(reference).dataobj) == CYST_LABEL
    dice = _dice(pred, ref)

    n_pred = detect_cyst_peaks(str(mask_path), PUBLISHED_PARAMS)
    n_ref = detect_cyst_peaks(str(reference), PUBLISHED_PARAMS)
    print(f"\nn5: {minutes:.1f} min, Dice(cyst) {dice:.4f}, "
          f"count {n_pred} predicted vs {n_ref} on the reference (not asserted)")

    assert dice >= 0.80


def test_predict_command_found_next_to_the_interpreter(tmp_path, monkeypatch):
    # A venv used without activation: nnUNetv2_predict sits next to sys.executable, not on PATH.
    from cystscan.segmentation import inference
    bindir = tmp_path / "venv" / "bin"; bindir.mkdir(parents=True)
    exe = bindir / "nnUNetv2_predict"; exe.write_text("#!/bin/sh\n"); exe.chmod(0o755)
    monkeypatch.setattr(inference.sys, "executable", str(bindir / "python"))
    monkeypatch.setenv("PATH", "/nonexistent")
    assert inference.predict_command() == str(exe)


def test_predict_command_falls_back_to_path(tmp_path, monkeypatch):
    from cystscan.segmentation import inference
    other = tmp_path / "bin"; other.mkdir()
    exe = other / "nnUNetv2_predict"; exe.write_text("#!/bin/sh\n"); exe.chmod(0o755)
    monkeypatch.setattr(inference.sys, "executable", str(tmp_path / "nobin" / "python"))
    monkeypatch.setenv("PATH", str(other))
    assert inference.predict_command() == str(exe)
