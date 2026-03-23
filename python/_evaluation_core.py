"""_evaluation_core.py
Shared logic for run_evaluation.py and run_evaluation_paper.py.
Not intended to be called directly.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure python/ is on the path
sys.path.insert(0, str(Path(__file__).parent))
from detect_cyst_peaks import detect_cyst_peaks


def _id_to_filename(sample_id: str) -> str:
    prefix = 'KRATS_' if sample_id[0] == 'K' else 'XRATS_'
    return f'{prefix}{int(sample_id[1:]):03d}.nii.gz'


def _compute_metrics(predicted: np.ndarray, gt: np.ndarray) -> dict:
    errors   = np.abs(predicted - gt)
    pct_err  = errors / gt
    MAPE = pct_err.mean() * 100
    MPE  = ((predicted - gt) / gt).mean() * 100
    MAE  = errors.mean()
    RMSE = np.sqrt((errors ** 2).mean())
    SS_res = ((gt - predicted) ** 2).sum()
    SS_tot = ((gt - gt.mean())   ** 2).sum()
    R2 = 1 - SS_res / SS_tot if SS_tot > 0 else float('nan')
    return {'MAPE': MAPE, 'MPE': MPE, 'MAE': MAE, 'RMSE': RMSE, 'R2': R2}


def run_evaluation_core(params, output_dir: Path, label: str = ''):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / 'data' / '2_Cyst_counting_pipeline' / 'Evaluation set'
    
    mask_dir = data_dir / 'Cyst masks'
    xls_file = data_dir / 'O1_O2_counts.xlsx'

    if not xls_file.is_file():
        sys.exit(f'Ground-truth file not found:\n  {xls_file}\nPlace the Zenodo data inside data/.')

    df = pd.read_excel(xls_file)
    samples  = [str(r) for r in df.iloc[:, 0]]
    gt_O1    = df.iloc[:, 1].to_numpy(dtype=float)
    gt_O2    = df.iloc[:, 2].to_numpy(dtype=float)
    gt_mean  = (gt_O1 + gt_O2) / 2
    n        = len(samples)
    predicted = np.zeros(n)

    print('\n' + '='*50)
    print(f'  EVALUATION  ({label})')
    print('='*50)
    print(f'Parameters: {np.round(params, 4).tolist()}\n')
    print(f'Processing {n} cases...\n')

    for i, sid in enumerate(samples):
        mask_path = mask_dir / _id_to_filename(sid)
        if not mask_path.is_file():
            sys.exit(f'Mask not found for {sid}:\n  {mask_path}')
        predicted[i] = detect_cyst_peaks(str(mask_path), params)
        print(f'  {sid} : predicted {int(predicted[i]):3d}  |  O1 {int(gt_O1[i]):3d}  |  O2 {int(gt_O2[i]):3d}')

    mO1 = _compute_metrics(predicted, gt_O1)
    mO2 = _compute_metrics(predicted, gt_O2)
    mMn = _compute_metrics(predicted, gt_mean)

    print(f'\n{"":20}  {"vs O1":>8}  {"vs O2":>8}  {"vs mean":>8}')
    for key, label_k in [('MAPE', 'MAPE (%)'), ('MPE', 'MPE (%)'),
                          ('MAE', 'MAE (cysts)'), ('RMSE', 'RMSE (cysts)'), ('R2', 'R²')]:
        print(f'  {label_k:<18}  {mO1[key]:>8.3f}  {mO2[key]:>8.3f}  {mMn[key]:>8.3f}')

    # Save CSV (column names match statistical_analysis.R expectations)
    csv_file = output_dir / 'evaluation_results.csv'
    out_df = pd.DataFrame({
        'Sample':     samples,
        'Operator 1': gt_O1.astype(int),
        'Operator 2': gt_O2.astype(int),
        'Auto':       predicted.astype(int),
    })
    out_df.to_csv(csv_file, index=False)
    print(f'\nResults saved to: {csv_file}')

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _scatter_panel(axes[0], gt_O1, predicted, mO1, 'Operator 1', color='#3366CC')
    _scatter_panel(axes[1], gt_O2, predicted, mO2, 'Operator 2', color='#CC3333')
    fig.suptitle(f'Evaluation set – predicted vs manual count ({label})', fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'scatter.png', dpi=150)
    fig.savefig(output_dir / 'scatter.svg')
    plt.close(fig)
    print(f'Scatter plot saved.\n')


def _scatter_panel(ax, gt, pred, m, gt_label, color):
    lo  = min(gt.min(), pred.min())
    hi  = max(gt.max(), pred.max())
    pad = (hi - lo) * 0.1
    axr = [lo - pad, hi + pad]
    p   = np.polyfit(gt, pred, 1)
    ax.scatter(gt, pred, s=70, alpha=0.85, color=color, zorder=5)
    ax.plot(axr, axr, 'k--', lw=1.5, label='Identity')
    xf = np.linspace(axr[0], axr[1], 100)
    ax.plot(xf, np.polyval(p, xf), 'r-', lw=1.5, label='Regression')
    ax.set_xlim(axr); ax.set_ylim(axr)
    ax.set_xlabel(f'{gt_label} (cysts)', fontweight='bold')
    ax.set_ylabel('Predicted (cysts)', fontweight='bold')
    ax.set_title(f'vs {gt_label}  (R²={m["R2"]:.3f}, MAPE={m["MAPE"]:.1f}%)')
    ax.legend(loc='lower right', fontsize=8); ax.grid(True); ax.set_aspect('equal')
