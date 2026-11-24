# Changelog

## 1.0.0 (unreleased)

### Changed

- **Python cyst counts now equal the MATLAB counts used in the paper.** On the
  10 Evaluation-set masks, with the published parameters, the Python counting
  gives the MATLAB counts exactly (before: differences up to 21 cysts, X18 264
  instead of 243). The Python steps were aligned with MATLAB: CLAHE is a port
  of `adapthisteq` (full output range, every slice), the Gaussian filter a port
  of `imgaussfilt3` (radius `ceil(2σ)`, replicate padding), the threshold uses
  the sample standard deviation, peaks are visited in column-major order, and
  the large-cyst merge compares every pair of peaks. Counts computed with
  earlier Python versions can differ from counts computed with this one.
- The code is an installable package, `cystscan`, with a command-line tool
  (`cystscan count`, `cystscan segment`). The old commands
  (`python python/run_*.py`, `python -m segmentation.run_inference`,
  `python -m segmentation.count_cysts`) still work after `pip install -e .`.
- A scan named `<case>_0000.nii.gz` now gives the case id `<case>` (the
  nnU-Net channel suffix is dropped), so its mask is `<case>.nii.gz`.
- Licence: Apache-2.0.

### Added

- The published parameters live in one file,
  `cystscan/data/published_params.json`, read by the Python and MATLAB scripts.
- Model weights are checked against the md5 listed in the Zenodo record; a
  corrupt archive is deleted and the download fails with an error.
- `python/run_optimization.py --seed` for repeatable optimisation runs.
- Test suite (pytest), synthetic test mask, CI workflow, CPU and GPU
  Dockerfiles with pinned dependency versions. The GPU image uses PyTorch
  2.14.1 with CUDA 12.6 (NVIDIA driver 560 or later).
- `cystscan segment` checks the model name and the scan (exists, 3-D) before
  any download, returns the mask path, and finds `nnUNetv2_predict` next to the
  Python interpreter, so a virtual environment works without activation.
- `test_segmentation_gpu`: optional end-to-end GPU check of `n5` on a scan
  and reference mask given by `CYSTSCAN_TEST_SCAN` and `CYSTSCAN_TEST_REFERENCE`.
