"""Write the synthetic demo mask used by the tests and by `cystscan count`.

    python tools/make_demo_fixtures.py --out tests/data

The mask is 96^3 voxels: a kidney ellipsoid (label 1) holding 12 separate
spherical cysts (label 2) with radii 3-8 voxels, placed with seed 0. The
script also records in expected_counts.json the count that
cystscan.counting.detect_cyst_peaks returns with the published parameters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

SHAPE = (96, 96, 96)
KIDNEY_SEMI_AXES = np.array([42.0, 32.0, 32.0])
N_CYSTS = 12
RADII = (3, 8)
GAP = 3  # voxels between sphere surfaces, so no two cysts touch (26-connectivity)


def synthetic_mask(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centre = (np.array(SHAPE) - 1) / 2
    grid = np.indices(SHAPE, dtype=np.float64)
    offs = grid - centre[:, None, None, None]
    kidney = (offs / KIDNEY_SEMI_AXES[:, None, None, None]) ** 2
    mask = (kidney.sum(axis=0) <= 1).astype(np.int16)

    spheres: list[tuple[np.ndarray, int]] = []
    while len(spheres) < N_CYSTS:
        r = int(rng.integers(RADII[0], RADII[1] + 1))
        # centre well inside the kidney: shrink the ellipsoid by r + 2
        u = rng.uniform(-1, 1, 3)
        if (u ** 2).sum() > 1:
            continue
        c = np.round(centre + u * (KIDNEY_SEMI_AXES - r - 2))
        if all(np.linalg.norm(c - c2) >= r + r2 + GAP for c2, r2 in spheres):
            spheres.append((c, r))

    for c, r in spheres:
        d2 = ((grid - c[:, None, None, None]) ** 2).sum(axis=0)
        mask[d2 <= r * r] = 2
    return mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("tests/data"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    path = args.out / "synthetic_mask.nii.gz"
    nib.save(nib.Nifti1Image(synthetic_mask(), np.eye(4)), str(path))

    from cystscan.counting import PUBLISHED_PARAMS, detect_cyst_peaks
    counts = {"synthetic_mask": detect_cyst_peaks(str(path), PUBLISHED_PARAMS)}
    (args.out / "expected_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
