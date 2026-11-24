"""cystscan.counting (formerly python/detect_cyst_peaks.py)
Count cysts in a NIfTI segmentation mask via distance-transform peak detection.

Python port of matlab/detectCystPeaks.m. It gives the same counts as the
MATLAB function: CLAHE and the Gaussian filter follow MATLAB's adapthisteq and
imgaussfilt3 (cystscan._matlab), voxels are visited in MATLAB's column-major
order, the threshold uses the sample standard deviation, and the large-cyst
merge compares every pair of peaks, as in MATLAB.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
from scipy.ndimage import label
from scipy.spatial.distance import cdist
from skimage.morphology import local_maxima

from cystscan._matlab import adapthisteq, imgaussfilt3
from cystscan.params import load_published_params
try:
    import edt as _edt
    def _distance_transform(mask):
        return _edt.edt(mask.astype(np.uint8), parallel=1).astype(np.float32)
except ImportError:
    from scipy.ndimage import distance_transform_edt as _scipy_edt
    def _distance_transform(mask):
        return _scipy_edt(mask).astype(np.float32)

# Published parameters (paper), from cystscan/data/published_params.json.
PUBLISHED_PARAMS = load_published_params()


def detect_cyst_peaks(mask_path: str, params: list | np.ndarray) -> int:
    """Count cysts in a segmentation mask.

    Parameters
    ----------
    mask_path : str
        Path to a NIfTI file (.nii or .nii.gz).
        Voxel label 2 = cyst; all other labels are ignored.
    params : sequence of 6 floats
        [minPeakSepSmall, minPeakSepLarge, largeCystVolThresh,
         maxPeakDistMerge, thresholdFraction, gaussianSigma]

    Returns
    -------
    int
        Number of cysts detected.
    """
    (min_sep_small, min_sep_large, vol_thresh,
     max_dist_merge, thresh_frac, gauss_sigma) = params

    # Load mask (a trailing singleton dimension is dropped, as in MATLAB)
    img = nib.load(mask_path)
    raw = np.squeeze(np.asarray(img.dataobj))
    cyst_mask = (raw == 2)

    if not cyst_mask.any():
        return 0

    # Distance transform (single precision, as bwdist), normalised to [0, 1]
    dt = _distance_transform(cyst_mask)
    dt /= dt.max()

    # Per-slice adaptive histogram equalisation (MATLAB adapthisteq defaults)
    for z in range(dt.shape[2]):
        dt[:, :, z] = adapthisteq(dt[:, :, z])

    # Gaussian smoothing and local maxima with 26-connectivity
    smoothed = imgaussfilt3(dt, gauss_sigma)
    del dt
    peak_markers = local_maxima(smoothed, connectivity=3, allow_borders=True)

    # Connected components (26-connectivity); voxels in column-major order,
    # like MATLAB's bwconncomp PixelIdxList.
    labeled, num_components = label(cyst_mask,
                                    structure=np.ones((3, 3, 3), dtype=np.int8))
    flat_labels   = labeled.ravel(order="F")
    flat_smoothed = smoothed.ravel(order="F")
    flat_peaks    = peak_markers.ravel(order="F")
    sort_order    = np.argsort(flat_labels, kind="stable")
    boundaries    = np.searchsorted(flat_labels[sort_order],
                                    np.arange(1, num_components + 2))

    num_cysts = 0

    for comp_id in range(num_components):
        flat_idx    = sort_order[boundaries[comp_id]:boundaries[comp_id + 1]]
        cyst_volume = flat_idx.size

        if cyst_volume > vol_thresh:
            min_sep  = min_sep_large
            do_merge = True
        else:
            min_sep  = min_sep_small
            do_merge = False

        region_dt    = flat_smoothed[flat_idx].astype(np.float64)
        region_peaks = flat_peaks[flat_idx]

        # Adaptive threshold: mean + fraction * sample std (MATLAB std)
        mu  = region_dt.mean()
        sig = region_dt.std(ddof=1) if cyst_volume > 1 else 0.0
        thr = mu + thresh_frac * sig

        valid_flat = flat_idx[region_peaks & (region_dt >= thr)]
        if valid_flat.size == 0:
            continue
        peak_coords = np.column_stack(
            np.unravel_index(valid_flat, cyst_mask.shape, order="F")
        ).astype(np.float64)

        # Suppress peaks that are too close together (greedy, in voxel order)
        D = cdist(peak_coords, peak_coords, metric="euclidean")
        np.fill_diagonal(D, np.inf)
        keep = np.ones(len(peak_coords), dtype=bool)
        for j in range(len(peak_coords)):
            if keep[j]:
                keep[D[j] < min_sep] = False
                keep[j] = True

        # Merge peaks within large cysts: as in MATLAB, a peak is dropped when
        # any earlier peak (suppressed or not) lies closer than max_dist_merge.
        if do_merge:
            close_to_earlier = np.triu(D < max_dist_merge, k=1).any(axis=0)
            keep &= ~close_to_earlier

        num_cysts += int(keep.sum())

    return num_cysts
