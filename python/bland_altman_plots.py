"""bland_altman_plots.py
Bland-Altman and linear regression plots for the evaluation results.

Reads  : results/evaluation/evaluation_results.csv
Writes : results/evaluation/plots/

This is a Python translation of statistical_analysis.R.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


def _bland_altman(ax, ref: np.ndarray, test: np.ndarray,
                  ref_label: str, test_label: str) -> dict:
    diff_pct  = (ref - test) / test * 100
    avg       = (ref + test) / 2
    bias      = diff_pct.mean()
    sd        = diff_pct.std(ddof=1)
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd
    y_range   = diff_pct.max() - diff_pct.min() if diff_pct.max() != diff_pct.min() else 1

    ax.scatter(avg, diff_pct, s=70, color='grey', edgecolors='black', linewidths=0.8, zorder=5)
    ax.axhline(0,         color='black', lw=1.5)
    ax.axhline(loa_upper, color='blue',  lw=1.5, ls='--')
    ax.axhline(bias,      color='red',   lw=1.5, ls='--')
    ax.axhline(loa_lower, color='blue',  lw=1.5, ls='--')

    x_ann = avg.max() * 0.93
    ax.text(x_ann, loa_upper - y_range * 0.08,
            f'+2×SD = {loa_upper:.3g}', color='blue',  fontsize=9, fontweight='bold', ha='right')
    ax.text(x_ann, bias      + y_range * 0.08,
            f'Bias = {bias:.3g}',       color='red',   fontsize=9, fontweight='bold', ha='right')
    ax.text(x_ann, loa_lower - y_range * 0.08,
            f'-2×SD = {loa_lower:.3g}', color='blue',  fontsize=9, fontweight='bold', ha='right')

    ax.set_title(f'{ref_label} vs {test_label}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Average number of cysts', fontsize=11, fontweight='bold')
    ax.set_ylabel('Difference (%)',           fontsize=11, fontweight='bold')
    ax.set_ylim(-30, 30)
    ax.grid(axis='y', color='#e0e0e0')
    ax.spines[['top', 'right']].set_visible(False)

    return {'bias': bias, 'sd': sd, 'loa_upper': loa_upper, 'loa_lower': loa_lower}


def _regression(ax, x: np.ndarray, y: np.ndarray,
                 x_label: str, y_label: str):
    slope, intercept, r, p, _ = stats.linregress(x, y)
    R2 = r ** 2
    p_str = 'P < 0.001' if p < 0.001 else f'P = {p:.3f}'
    eq = f'y = {slope:.2f}x + {intercept:.2f}\nR² = {R2:.3f}\n{p_str}'

    ax.scatter(x, y, s=60, color='#009E73', alpha=0.85, zorder=5)
    xf = np.linspace(x.min(), x.max(), 100)
    yf = slope * xf + intercept
    # Confidence band
    se = np.std(y - (slope * x + intercept), ddof=2)
    ci = 1.96 * se * np.sqrt(1/len(x) + (xf - x.mean())**2 / ((x - x.mean())**2).sum())
    ax.fill_between(xf, yf - ci, yf + ci, color='grey', alpha=0.2)
    ax.plot(xf, yf, 'k-', lw=1.5)

    xr = x.max() - x.min(); yr = y.max() - y.min()
    ax.text(x.min() + 0.05*xr, y.max() - 0.05*yr, eq,
            va='top', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', fc='white', ec='black', lw=0.8))

    ax.set_title(f'{x_label} vs {y_label}', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'{x_label} (number of cysts)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{y_label} (number of cysts)', fontsize=11, fontweight='bold')
    ax.grid(color='#e0e0e0'); ax.spines[['top', 'right']].set_visible(False)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    csv_file  = repo_root / 'results' / 'evaluation' / 'evaluation_results.csv'
    plot_dir  = repo_root / 'results' / 'evaluation' / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.is_file():
        sys.exit(
            f'Results file not found:\n  {csv_file}\n'
            'Run run_evaluation.py (or run_evaluation_paper.py) first.'
        )

    df  = pd.read_csv(csv_file)
    O1  = df['Operator 1'].to_numpy(dtype=float)
    O2  = df['Operator 2'].to_numpy(dtype=float)
    Auto = df['Auto'].to_numpy(dtype=float)

    # ---- Bland-Altman plots -----------------------------------------------
    print('Generating Bland-Altman plots...')
    pairs = [
        (O1, O2,   'Operator 1', 'Operator 2', 'bland_altman_op1_op2.png'),
        (O1, Auto, 'Operator 1', 'Auto',        'bland_altman_op1_auto.png'),
        (O2, Auto, 'Operator 2', 'Auto',        'bland_altman_op2_auto.png'),
    ]
    for ref, test, rl, tl, fname in pairs:
        fig, ax = plt.subplots(figsize=(6, 4))
        res = _bland_altman(ax, ref, test, rl, tl)
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=300)
        plt.close(fig)
        print(f'  {rl} vs {tl}:  Bias={res["bias"]:.3g}%  SD={res["sd"]:.3g}%  '
              f'LoA=[{res["loa_lower"]:.3g}%, {res["loa_upper"]:.3g}%]')

    # ---- Regression plots -------------------------------------------------
    print('\nGenerating regression plots...')
    reg_pairs = [
        (O1, O2,   'Operator 1', 'Operator 2', 'regression_op1_op2.png'),
        (O1, Auto, 'Operator 1', 'Auto',        'regression_op1_auto.png'),
        (O2, Auto, 'Operator 2', 'Auto',        'regression_op2_auto.png'),
    ]
    for x, y, xl, yl, fname in reg_pairs:
        fig, ax = plt.subplots(figsize=(6, 4))
        _regression(ax, x, y, xl, yl)
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=300, facecolor='white')
        plt.close(fig)
        print(f'  {xl} vs {yl}: saved')

    print(f'\nAll plots saved to: {plot_dir}\n')


if __name__ == '__main__':
    main()
