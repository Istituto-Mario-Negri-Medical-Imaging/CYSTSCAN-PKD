"""sensitivity_analysis.py
Three-phase sensitivity analysis of the cyst-counting parameters.

  Phase 1 – Morris screening (SALib implementation)
  Phase 2 – One-at-a-time (OAT) analysis, all 6 parameters ± 30 %
  Phase 3 – 2-D parameter interaction for the top 2 parameters

This is a Python translation of sensitivity_analysis.m.
Phase 1 uses SALib instead of a manual trajectory implementation;
results are methodologically equivalent.
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fitness_function import evaluate_params

# ---- Configuration --------------------------------------------------------

OPTIMAL_PARAMS = np.array([
    4.27269938837114,
    18.2632731580751,
    42144.3492985436,
    14.9647743085524,
    0.111299008136904,
    0.918315428393029,
])
PARAM_NAMES = ['minPeakSepSmall', 'minPeakSepLarge', 'largeCystVolThresh',
               'maxPeakDistMerge', 'thresholdFraction', 'gaussianSigma']
LB = OPTIMAL_PARAMS * 0.7
UB = OPTIMAL_PARAMS * 1.3

BATCH_SIZE = 8


def _id_to_filename(sample_id: str) -> str:
    prefix = 'KRATS_' if sample_id[0] == 'K' else 'XRATS_'
    return f'{prefix}{int(sample_id[1:]):03d}.nii.gz'


def _eval_batch(param_sets, gt_masks, gt_counts, n_workers=None):
    """Evaluate a list of parameter sets; return (MAPE_array, MAE_array)."""
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    scores = []
    maes   = []
    n  = len(param_sets)
    nB = (n + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(nB):
        batch = param_sets[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        for p in batch:
            s, m = evaluate_params(p, gt_masks, gt_counts, n_workers=n_workers)
            scores.append(s)
            maes.append(m['MAE'])
        print(f'  batch {b+1}/{nB} done')
    return np.array(scores), np.array(maes)


def run_sensitivity_analysis():
    repo_root  = Path(__file__).resolve().parent.parent
    data_dir   = repo_root / 'data' / 'Optimization set'
    mask_dir   = data_dir / 'Cyst masks'
    xls_file   = data_dir / 'O1_count.xlsx'
    output_dir = repo_root / 'results' / 'sensitivity_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not xls_file.is_file():
        sys.exit(f'Ground-truth file not found:\n  {xls_file}')

    df = pd.read_excel(xls_file)
    gt_masks  = []
    gt_counts = []
    for _, row in df.iterrows():
        sid  = str(row.iloc[0])
        path = mask_dir / _id_to_filename(sid)
        if not path.is_file():
            sys.exit(f'Mask not found for {sid}:\n  {path}')
        gt_masks.append(str(path))
        gt_counts.append(int(row.iloc[1]))
    gt_counts = np.array(gt_counts)

    # ---- Phase 1: Morris screening ----------------------------------------
    print('\n=== Phase 1: Morris screening ===')
    t0 = time.time()

    try:
        from SALib.sample import morris as morris_sample
        from SALib.analyze import morris as morris_analyze
    except ImportError:
        sys.exit('SALib is required: pip install SALib')

    problem = {
        'num_vars': 6,
        'names':    PARAM_NAMES,
        'bounds':   list(zip(LB.tolist(), UB.tolist())),
    }

    # r=20 trajectories, matches MATLAB
    X_norm = morris_sample.sample(problem, N=20, num_levels=6,
                                  optimal_trajectories=None, seed=42)
    print(f'Morris sample: {X_norm.shape[0]} evaluations')

    param_sets = [X_norm[i] for i in range(X_norm.shape[0])]
    scores_m, _ = _eval_batch(param_sets, gt_masks, gt_counts, n_workers=1)

    Si = morris_analyze.analyze(problem, X_norm, scores_m, num_resamples=100, seed=42)

    mu_star = Si['mu_star']
    mu      = Si['mu']
    sigma   = Si['sigma']

    print(f'\nMorris completed in {(time.time()-t0)/60:.1f} min')
    print(f'\n{"Parameter":<25}  {"mu*":>8}  {"mu":>8}  {"sigma":>8}')
    print('-' * 55)
    sort_idx = np.argsort(mu_star)[::-1]
    thr = np.median(mu_star)
    for i in sort_idx:
        star = ' *' if mu_star[i] > thr else ''
        print(f'{PARAM_NAMES[i]:<25}  {mu_star[i]:>8.4f}  {mu[i]:>8.4f}  {sigma[i]:>8.4f}{star}')

    # Morris plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mu_star, sigma, s=100, alpha=0.8)
    for i, name in enumerate(PARAM_NAMES):
        ax.annotate(name, (mu_star[i]*1.03, sigma[i]*1.03), fontsize=9)
    ax.axvline(thr, color='r', ls='--', lw=1.5)
    ax.axhline(np.median(sigma), color='b', ls='--', lw=1.5)
    ax.set_xlabel('μ* (mean absolute elementary effect)', fontweight='bold')
    ax.set_ylabel('σ (standard deviation)', fontweight='bold')
    ax.set_title('Morris screening', fontweight='bold')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / 'morris_screening.png', dpi=150)
    plt.close(fig)

    # ---- Phase 2: OAT analysis --------------------------------------------
    print('\n=== Phase 2: One-at-a-time analysis ===')
    t1 = time.time()

    variations = np.linspace(-0.3, 0.3, 21)
    oat_param_sets = []
    param_map      = []
    for rank, pi in enumerate(sort_idx):
        for v_idx, v in enumerate(variations):
            tp = OPTIMAL_PARAMS.copy()
            tp[pi] *= (1 + v)
            oat_param_sets.append(tp)
            param_map.append((rank, v_idx))

    print(f'Evaluating {len(oat_param_sets)} parameter combinations...')
    oat_scores, oat_maes = _eval_batch(oat_param_sets, gt_masks, gt_counts, n_workers=1)
    print(f'OAT completed in {(time.time()-t1)/60:.1f} min')

    oat_results = []
    for rank in range(len(sort_idx)):
        idxs = [i for i, (r, _) in enumerate(param_map) if r == rank]
        oat_results.append({
            'param':      PARAM_NAMES[sort_idx[rank]],
            'variations': variations,
            'scores':     oat_scores[idxs],
            'MAEs':       oat_maes[idxs],
        })

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for rank, (d, ax) in enumerate(zip(oat_results, axes.flat)):
        ax2 = ax.twinx()
        ax.plot(d['variations']*100, d['scores'], 'o-', lw=2, ms=6, color='#3366CC')
        ax2.plot(d['variations']*100, d['MAEs'],  's--', lw=2, ms=6, color='#CC3333')
        ax.set_xlabel('Variation (%)')
        ax.set_ylabel('MAPE (%)', color='#3366CC')
        ax2.set_ylabel('MAE (cysts)', color='#CC3333')
        ax.axvline(0, color='r', ls='--', lw=2)
        ax.set_title(f'{d["param"]} (rank {rank+1})', fontweight='bold')
        ax.grid(True)
    fig.suptitle('One-at-a-time sensitivity analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'oat_analysis.png', dpi=150)
    plt.close(fig)

    # ---- Phase 3: 2-D interaction -----------------------------------------
    print('\n=== Phase 3: 2-D parameter interaction ===')
    t2 = time.time()

    p1_idx = sort_idx[0]
    p2_idx = sort_idx[1]
    grid_size = 9
    grid_vals = np.linspace(-0.25, 0.25, grid_size)

    grid_sets = []
    grid_map  = []
    for i, vi in enumerate(grid_vals):
        for j, vj in enumerate(grid_vals):
            tp = OPTIMAL_PARAMS.copy()
            tp[p1_idx] *= (1 + vi)
            tp[p2_idx] *= (1 + vj)
            grid_sets.append(tp)
            grid_map.append((i, j))

    print(f'Computing {grid_size}×{grid_size} interaction grid...')
    g_scores, g_maes = _eval_batch(grid_sets, gt_masks, gt_counts, n_workers=1)
    print(f'2-D interaction completed in {(time.time()-t2)/60:.1f} min')

    int_scores = np.zeros((grid_size, grid_size))
    int_maes   = np.zeros((grid_size, grid_size))
    for k, (i, j) in enumerate(grid_map):
        int_scores[i, j] = g_scores[k]
        int_maes[i, j]   = g_maes[k]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, title in [(axes[0], int_scores, 'MAPE landscape (%)'),
                             (axes[1], int_maes,   'MAE landscape (cysts)')]:
        im = ax.imshow(data, extent=[grid_vals[0]*100, grid_vals[-1]*100,
                                     grid_vals[-1]*100, grid_vals[0]*100],
                       aspect='auto', origin='upper', cmap='jet')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(f'{PARAM_NAMES[p2_idx]} variation (%)', fontweight='bold')
        ax.set_ylabel(f'{PARAM_NAMES[p1_idx]} variation (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.plot(0, 0, 'wx', ms=15, mew=3)
    fig.suptitle(f'Parameter interaction: {PARAM_NAMES[p1_idx]} vs {PARAM_NAMES[p2_idx]}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'interaction_2d.png', dpi=150)
    plt.close(fig)

    # ---- Save results ------------------------------------------------------
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.savez(output_dir / f'sensitivity_analysis_{ts}.npz',
             mu_star=mu_star, mu=mu, sigma=sigma,
             sort_idx=sort_idx, param_names=PARAM_NAMES,
             optimal_params=OPTIMAL_PARAMS)
    print(f'\nResults saved to: {output_dir}\n')


if __name__ == '__main__':
    run_sensitivity_analysis()
