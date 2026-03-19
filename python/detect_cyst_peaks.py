"""detect_cyst_peaks.py
Count cysts in a NIfTI segmentation mask via distance-transform peak detection.

This is a direct Python translation of detectCystPeaks.m.
See python/README.md for known differences from the MATLAB implementation.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, label
from scipy.spatial.distance import cdist
from skimage.exposure import equalize_adapthist
from skimage.morphology import local_maxima
try:
    import edt as _edt
    def _distance_transform(mask):
        return _edt.edt(mask.astype(np.uint8), parallel=1).astype(np.float32)
except ImportError:
    from scipy.ndimage import distance_transform_edt as _scipy_edt
    def _distance_transform(mask):
        return _scipy_edt(mask).astype(np.float32)


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

    # Load mask
    img = nib.load(mask_path)
    raw = np.squeeze(np.asarray(img.dataobj, dtype=np.int16))
    cyst_mask = (raw == 2)

    if not cyst_mask.any():
        return 0

    # Distance transform, normalised to [0, 1]
    dt = _distance_transform(cyst_mask)
    dt_max = dt.max()
    if dt_max > 0:
        dt /= dt_max

    # Per-slice adaptive histogram equalisation (CLAHE)
    # MATLAB's adapthisteq with Range='original' (default) maps output back to
    # the input slice's own [min, max] range.  skimage's equalize_adapthist
    # always outputs [0, 1], so we rescale explicitly to match MATLAB.
    for z in range(dt.shape[2]):
        slc = dt[:, :, z]
        slc_min, slc_max = float(slc.min()), float(slc.max())
        if slc_max > slc_min:
            clahe = equalize_adapthist(slc)          # → [0, 1]
            dt[:, :, z] = (clahe * (slc_max - slc_min) + slc_min).astype(np.float32)

    # Gaussian smoothing
    smoothed = gaussian_filter(dt, sigma=gauss_sigma)

    # Local maxima with 26-connectivity (3x3x3 neighbourhood)
    peak_markers = local_maxima(smoothed, connectivity=3)

    # Connected component analysis
    labeled, num_components = label(cyst_mask,
                                    structure=np.ones((3, 3, 3), dtype=np.int8))

    # Pre-build flat pixel-index lists for each region (mirrors MATLAB's cc.PixelIdxList).
    # This avoids scanning the full volume once per component.
    flat_labels   = labeled.ravel()
    flat_smoothed = smoothed.ravel()
    flat_peaks    = peak_markers.ravel()
    sort_order    = np.argsort(flat_labels, kind='stable')
    sorted_labels = flat_labels[sort_order]
    boundaries    = np.searchsorted(sorted_labels, np.arange(1, num_components + 2))

    num_cysts = 0

    for comp_id in range(num_components):
        flat_idx    = sort_order[boundaries[comp_id]:boundaries[comp_id + 1]]
        cyst_volume = flat_idx.size

        if cyst_volume == 0:
            continue

        if cyst_volume > vol_thresh:
            min_sep  = min_sep_large
            do_merge = True
        else:
            min_sep  = min_sep_small
            do_merge = False

        # Local distance values and peak flags for this region
        region_dt    = flat_smoothed[flat_idx]
        region_peaks = flat_peaks[flat_idx]

        # Adaptive threshold
        mu  = region_dt.mean()
        sig = region_dt.std()
        thr = mu + thresh_frac * sig

        # Valid peaks: must be a regional maximum and above threshold
        valid_mask  = region_peaks & (region_dt >= thr)
        # Convert flat indices to 3D coordinates for distance computation
        valid_flat  = flat_idx[valid_mask]
        peak_coords = np.column_stack(np.unravel_index(valid_flat, cyst_mask.shape))  # (K,3)

        if peak_coords.shape[0] == 0:
            continue

        # Suppress nearby peaks (greedy, same logic as MATLAB)
        D = cdist(peak_coords, peak_coords, metric='euclidean')
        np.fill_diagonal(D, np.inf)

        keep = np.ones(len(peak_coords), dtype=bool)
        for j in range(len(peak_coords)):
            if keep[j]:
                too_close = D[j] < min_sep
                keep[too_close] = False
                keep[j] = True

        # Merge peaks within large cysts
        if do_merge:
            surviving = np.where(keep)[0]
            for j_idx, j in enumerate(surviving):
                for k in surviving[j_idx + 1:]:
                    if keep[j] and keep[k]:
                        if np.linalg.norm(peak_coords[j] - peak_coords[k]) < max_dist_merge:
                            keep[k] = False

        num_cysts += int(keep.sum())

    return num_cysts
