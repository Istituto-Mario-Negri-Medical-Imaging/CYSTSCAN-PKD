"""fitness_function.py
Evaluate a parameter set against ground-truth cyst counts.

This is a direct Python translation of fitness_function.m.
"""

from __future__ import annotations
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from detect_cyst_peaks import detect_cyst_peaks


def _worker(args):
    mask_path, params = args
    return detect_cyst_peaks(mask_path, params)


def evaluate_params(
    params: list | np.ndarray,
    gt_masks: list[str],
    gt_counts: list[int] | np.ndarray,
    n_workers: int = -1,
) -> tuple[float, dict]:
    """Evaluate a parameter set against all training masks.

    Parameters
    ----------
    params    : 6-element sequence – see detect_cyst_peaks for definitions.
    gt_masks  : list of paths to NIfTI mask files.
    gt_counts : ground-truth cyst counts (same order as gt_masks).
    n_workers : number of parallel workers (-1 = all CPUs).

    Returns
    -------
    score   : MAPE (%) – value to minimise.
    metrics : dict with per-case and aggregate statistics.
    """
    gt = np.asarray(gt_counts, dtype=float)
    n  = len(gt_masks)

    import os
    if n_workers == -1:
        n_workers = os.cpu_count() or 1

    args = [(m, params) for m in gt_masks]
    if n_workers == 1:
        # Sequential: avoids spawning subprocesses that each load large NIfTI files
        predictions = np.array([_worker(a) for a in args], dtype=float)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            predictions = np.array(list(pool.map(_worker, args)), dtype=float)

    errors       = np.abs(predictions - gt)
    pct_error    = (predictions - gt) / gt
    abs_pct_err  = np.abs(pct_error)
    overest      = np.maximum(0, predictions - gt)
    underest     = np.maximum(0, gt - predictions)

    MAPE      = abs_pct_err.mean() * 100
    MPE       = pct_error.mean()   * 100
    bias_pen  = max(0, abs(MPE) - 10) * 2
    MAE       = errors.mean()
    STD       = errors.std(ddof=1)
    RMSE      = np.sqrt((errors ** 2).mean())
    MAD       = np.median(errors)
    SS_res    = ((gt - predictions) ** 2).sum()
    SS_tot    = ((gt - gt.mean())   ** 2).sum()
    R2        = 1 - SS_res / SS_tot if SS_tot > 0 else float('nan')
    mean_bias = (predictions - gt).mean()
    ratio_over = overest.sum() / (underest.sum() + 1e-6)

    score = MAPE   # Option A (MAPE only) – matches MATLAB default

    metrics = {
        'MAPE':         MAPE,
        'MPE':          MPE,
        'MAE':          MAE,
        'STD':          STD,
        'RMSE':         RMSE,
        'MAD':          MAD,
        'R_squared':    R2,
        'mean_bias':    mean_bias,
        'ratio_over':   ratio_over,
        'predictions':  predictions,
        'ground_truth': gt,
        'errors':       errors,
        'score_components': {'primary': MAPE, 'bias_penalty': bias_pen},
    }
    return score, metrics
