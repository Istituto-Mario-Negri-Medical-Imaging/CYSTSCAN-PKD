"""Python re-implementations of the MATLAB Image Processing Toolbox steps
used by ``detectCystPeaks.m``, written so that the Python counts equal the
MATLAB counts published with the paper.

Behaviour follows the MATLAB documentation and observed outputs:

* ``adapthisteq`` (CLAHE): default options (NumTiles [8 8], ClipLimit 0.01,
  NBins 256, Distribution 'uniform', Range 'full'), symmetric padding to even
  tile sizes, MATLAB's clip-and-redistribute rule, bilinear interpolation of
  the tile mappings, single-precision output.
* ``imgaussfilt3``: separable kernel of size ``2*ceil(2*sigma)+1``,
  replicate padding, filtered along columns, rows, then slices.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import correlate1d


def _round_half_away(x: float) -> int:
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def _clip_histogram(hist: np.ndarray, clip_limit: int, n_bins: int) -> np.ndarray:
    """MATLAB adapthisteq clip limit with its redistribution order."""
    hist = hist.astype(np.int64).copy()
    total_excess = int(np.maximum(hist - clip_limit, 0).sum())
    avg_incr = total_excess // n_bins
    upper = clip_limit - avg_incr

    over = hist > clip_limit
    high = ~over & (hist > upper)
    low = ~over & ~high
    total_excess -= int((clip_limit - hist[high]).sum())
    total_excess -= avg_incr * int(low.sum())
    hist[over | high] = clip_limit
    hist[low] += avg_incr

    k = 0
    while total_excess != 0:
        step = max(n_bins // total_excess, 1)
        for m in range(k, n_bins, step):
            if hist[m] < clip_limit:
                hist[m] += 1
                total_excess -= 1
                if total_excess == 0:
                    break
        k += 1
        if k >= n_bins:
            k = 0
    return hist


def adapthisteq(image: np.ndarray, num_tiles=(8, 8), clip_limit: float = 0.01,
                n_bins: int = 256) -> np.ndarray:
    """CLAHE on a 2-D floating-point image in [0, 1], as MATLAB ``adapthisteq``.

    Returns a float32 array of the same shape.
    """
    img = np.asarray(image, dtype=np.float32)
    rows, cols = img.shape
    nt_r, nt_c = num_tiles

    row_div = rows % nt_r == 0
    col_div = cols % nt_c == 0
    pad = not (row_div and col_div
               and (rows // nt_r) % 2 == 0 and (cols // nt_c) % 2 == 0)
    pre_r = pre_c = 0
    if pad:
        if row_div:
            tile_r, pad_r = rows // nt_r, 0
        else:
            tile_r = rows // nt_r + 1
            pad_r = tile_r * nt_r - rows
        if col_div:
            tile_c, pad_c = cols // nt_c, 0
        else:
            tile_c = cols // nt_c + 1
            pad_c = tile_c * nt_c - cols
        if tile_r % 2:
            pad_r += nt_r
        if tile_c % 2:
            pad_c += nt_c
        pre_r, pre_c = pad_r // 2, pad_c // 2
        img = np.pad(img, ((pre_r, pad_r - pre_r), (pre_c, pad_c - pre_c)),
                     mode="symmetric")

    dim_r, dim_c = img.shape[0] // nt_r, img.shape[1] // nt_c
    n_pix = dim_r * dim_c
    min_clip = math.ceil(n_pix / n_bins)
    clip = min_clip + _round_half_away(clip_limit * (n_pix - min_clip))

    # Grey level of each pixel on the n_bins scale (imhist / grayxform rule).
    idx = np.floor(img.astype(np.float64) * (n_bins - 1) + 0.5)
    idx = np.clip(np.nan_to_num(idx, nan=0.0), 0, n_bins - 1).astype(np.intp)

    # Tile mappings, cached by histogram (background tiles repeat).
    scale = 1.0 / n_pix
    cache: dict[bytes, np.ndarray] = {}
    maps = np.empty((nt_r, nt_c, n_bins), dtype=np.float32)
    for r in range(nt_r):
        for c in range(nt_c):
            tile = idx[r * dim_r:(r + 1) * dim_r, c * dim_c:(c + 1) * dim_c]
            hist = np.bincount(tile.ravel(), minlength=n_bins)
            key = hist.tobytes()
            m = cache.get(key)
            if m is None:
                h = _clip_histogram(hist, clip, n_bins)
                m = np.minimum(np.cumsum(h) * scale, 1.0).astype(np.float32)
                cache[key] = m
            maps[r, c] = m

    # Bilinear interpolation between the four neighbouring tile mappings.
    out = np.empty(img.shape, dtype=np.float32)
    half_r, half_c = dim_r // 2, dim_c // 2
    r0 = 0
    for k in range(nt_r + 1):
        if k == 0:
            n_r, mr = half_r, (0, 0)
        elif k == nt_r:
            n_r, mr = half_r, (nt_r - 1, nt_r - 1)
        else:
            n_r, mr = dim_r, (k - 1, k)
        c0 = 0
        for col in range(nt_c + 1):
            if col == 0:
                n_c, mc = half_c, (0, 0)
            elif col == nt_c:
                n_c, mc = half_c, (nt_c - 1, nt_c - 1)
            else:
                n_c, mc = dim_c, (col - 1, col)
            block = idx[r0:r0 + n_r, c0:c0 + n_c]
            ul = maps[mr[0], mc[0]][block].astype(np.float64)
            ur = maps[mr[0], mc[1]][block].astype(np.float64)
            bl = maps[mr[1], mc[0]][block].astype(np.float64)
            br = maps[mr[1], mc[1]][block].astype(np.float64)
            row_w = np.arange(n_r, dtype=np.float64)[:, None]
            col_w = np.arange(n_c, dtype=np.float64)[None, :]
            row_rev = np.arange(n_r, 0, -1, dtype=np.float64)[:, None]
            col_rev = np.arange(n_c, 0, -1, dtype=np.float64)[None, :]
            val = (row_rev * (col_rev * ul + col_w * ur)
                   + row_w * (col_rev * bl + col_w * br)) / (n_r * n_c)
            out[r0:r0 + n_r, c0:c0 + n_c] = val
            c0 += n_c
        r0 += n_r

    return out[pre_r:pre_r + rows, pre_c:pre_c + cols]


def gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = math.ceil(2 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    h = np.exp(-(x * x) / (sigma * sigma) / 2)
    h[h < np.finfo(np.float64).eps * h.max()] = 0
    return h / h.sum()


def imgaussfilt3(volume: np.ndarray, sigma: float) -> np.ndarray:
    """3-D Gaussian smoothing as MATLAB ``imgaussfilt3(V, sigma)`` (float32)."""
    h = gaussian_kernel_1d(sigma)
    out = np.asarray(volume, dtype=np.float32)
    for axis in (1, 0, 2):
        out = correlate1d(out, h, axis=axis, mode="nearest", output=np.float32)
    return out
