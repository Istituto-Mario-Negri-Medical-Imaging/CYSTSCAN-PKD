"""run_optimization.py
Genetic algorithm optimisation of cyst-counting parameters using DEAP.

Reads the Optimization set from data/, runs the GA, and writes results
to results/optimization/.

This is a Python translation of run_optimization.m.
See python/README.md for differences from the MATLAB implementation.
"""

import os
import sys
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

# Ensure the python/ directory is on the path when called from elsewhere
sys.path.insert(0, str(Path(__file__).parent))
from fitness_function import evaluate_params


# ---- Parameter bounds (must match MATLAB) ---------------------------------
PARAM_NAMES = ['minPeakSepSmall', 'minPeakSepLarge', 'largeCystVolThresh',
               'maxPeakDistMerge', 'thresholdFraction', 'gaussianSigma']
LB = np.array([1,  10,  10000, 1,  0.1, 0.01])
UB = np.array([15, 100, 150000, 15, 3.0, 1.5])

# ---- GA hyperparameters (mirror MATLAB) -----------------------------------
POP_SIZE     = 50
N_GEN        = 200
ELITE_COUNT  = 2
CX_PROB      = 0.5      # probability of crossover per pair
MUT_PROB     = 0.95     # probability of mutation per individual
TOURNSIZE    = 3
STALL_LIMIT  = 5        # generations without improvement before diversity injection
REINIT_FRAC  = 0.5      # fraction of non-elite population reinitialised on stall


def _eval_individual(ind, gt_masks, gt_counts):
    """Evaluate one individual sequentially (no subprocesses – mirrors MATLAB's UseParallel=false)."""
    try:
        if any(not np.isfinite(v) or v <= 0 for v in ind):
            return (1e6,)
        score, _ = evaluate_params(ind, gt_masks, gt_counts, n_workers=1)
        return (score if np.isfinite(score) and score >= 0 else 1e6,)
    except Exception:
        return (1e6,)


def _id_to_filename(sample_id: str) -> str:
    prefix = 'KRATS_' if sample_id[0] == 'K' else 'XRATS_'
    return f'{prefix}{int(sample_id[1:]):03d}.nii.gz'


def _clip(individual):
    """Clip individual to parameter bounds in-place."""
    for i, (lo, hi) in enumerate(zip(LB, UB)):
        individual[i] = float(np.clip(individual[i], lo, hi))
    return individual


def run_optimization():
    repo_root  = Path(__file__).resolve().parent.parent
    data_dir   = repo_root / 'data' / '2_Cyst_counting_pipeline' / 'Optimization set'
    mask_dir   = data_dir / 'Cyst masks'
    output_dir = repo_root / 'results' / 'optimization'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ground truth ------------------------------------------------
    xls_file = data_dir / 'O1_count.xlsx'
    if not xls_file.is_file():
        sys.exit(f'Ground-truth file not found:\n  {xls_file}\nPlace the Zenodo data inside data/.')

    df = pd.read_excel(xls_file)
    gt_masks  = []
    gt_counts = []
    for _, row in df.iterrows():
        sid  = str(row.iloc[0])
        cnt  = int(row.iloc[1])
        path = mask_dir / _id_to_filename(sid)
        if not path.is_file():
            sys.exit(f'Mask not found for {sid}:\n  {path}')
        gt_masks.append(str(path))
        gt_counts.append(cnt)

    gt_counts = np.array(gt_counts)

    print('\n' + '='*50)
    print('  GENETIC ALGORITHM OPTIMISATION')
    print('='*50)
    print(f'Training samples : {len(gt_counts)}')
    print(f'Count range      : [{gt_counts.min()} – {gt_counts.max()}]  '
          f'(mean {gt_counts.mean():.1f} ± {gt_counts.std():.1f})\n')

    # ---- DEAP setup -------------------------------------------------------
    # Minimisation problem
    if hasattr(creator, 'FitnessMin'):
        del creator.FitnessMin
    if hasattr(creator, 'Individual'):
        del creator.Individual
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    def random_individual():
        ind = [float(LB[i] + random.random() * (UB[i] - LB[i])) for i in range(6)]
        return creator.Individual(ind)

    toolbox.register('individual', random_individual)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    def eval_individual(ind):
        return _eval_individual(ind, gt_masks, gt_counts)

    toolbox.register('evaluate', eval_individual)
    toolbox.register('select',   tools.selTournament, tournsize=TOURNSIZE)
    # cxBlend is the closest available equivalent to MATLAB's adaptive feasible crossover
    toolbox.register('mate',     tools.cxBlend, alpha=0.5)
    # mutPolynomialBounded is the closest equivalent to MATLAB's adaptive feasible mutation
    toolbox.register('mutate',   tools.mutPolynomialBounded,
                     low=LB.tolist(), up=UB.tolist(),
                     eta=20.0, indpb=1.0 / 6)

    # ---- Initialise population --------------------------------------------
    population = toolbox.population(n=POP_SIZE)
    hof        = tools.HallOfFame(ELITE_COUNT)

    # Evaluate initial population sequentially (mirrors MATLAB's UseParallel=false)
    print('Evaluating initial population...', flush=True)
    for i, ind in enumerate(population):
        ind.fitness.values = eval_individual(ind)
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{POP_SIZE} individuals evaluated', flush=True)

    hof.update(population)

    # ---- History tracking -------------------------------------------------
    history = {
        'generation': [], 'best_score': [], 'MPE': [], 'MAE': [],
        'STD': [], 'R2': [], 'mean_score': [],
    }

    best_ever  = hof[0].fitness.values[0]
    stall_gen  = 0

    print(f'{"Gen":>6}  {"Best MAPE":>12}  {"MPE":>8}  {"MAE±STD":>14}  {"R²":>8}  {"Stall":>6}')
    print('-' * 70)

    # ---- Main GA loop -----------------------------------------------------
    for gen in range(1, N_GEN + 1):
        # Selection (exclude elite slots)
        elite    = tools.selBest(population, ELITE_COUNT)
        offspring = toolbox.select(population, POP_SIZE - ELITE_COUNT)
        offspring = [toolbox.clone(o) for o in offspring]

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                _clip(mutant)
                del mutant.fitness.values

        # Evaluate new individuals sequentially
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = eval_individual(ind)

        # Combine elite + offspring
        population[:] = list(elite) + offspring
        hof.update(population)

        # Stats
        best_score = hof[0].fitness.values[0]
        valid_scores = [ind.fitness.values[0] for ind in population
                        if np.isfinite(ind.fitness.values[0]) and ind.fitness.values[0] < 1e5]
        mean_score = np.mean(valid_scores) if valid_scores else float('nan')

        _, m = evaluate_params(list(hof[0]), gt_masks, gt_counts, n_workers=1)

        if best_score < best_ever - 1e-6:
            best_ever = best_score
            stall_gen = 0
            print(f'{gen:>6}  {best_score:>12.4f}  {m["MPE"]:>8.2f}  '
                  f'{m["MAE"]:>6.2f}±{m["STD"]:<5.2f}  {m["R_squared"]:>8.4f}')
        else:
            stall_gen += 1
            print(f'{gen:>6}  {best_score:>12.4f}  {m["MPE"]:>8.2f}  '
                  f'{m["MAE"]:>6.2f}±{m["STD"]:<5.2f}  {m["R_squared"]:>8.4f}  {stall_gen:>6}')

        # Diversity injection
        if stall_gen >= STALL_LIMIT:
            print(f'   >>> Diversity injection: reinitialising {int(REINIT_FRAC*100)}% of population <<<')
            n_reinit = round(REINIT_FRAC * (POP_SIZE - ELITE_COUNT))
            reinit_idx = random.sample(range(ELITE_COUNT, POP_SIZE), n_reinit)
            for idx in reinit_idx:
                population[idx] = random_individual()
                population[idx].fitness.values = eval_individual(population[idx])
            stall_gen = 0

        history['generation'].append(gen)
        history['best_score'].append(best_score)
        history['MPE'].append(m['MPE'])
        history['MAE'].append(m['MAE'])
        history['STD'].append(m['STD'])
        history['R2'].append(m['R_squared'])
        history['mean_score'].append(mean_score)

    # ---- Final results ----------------------------------------------------
    best_params = list(hof[0])
    best_score  = hof[0].fitness.values[0]
    _, final_m  = evaluate_params(best_params, gt_masks, gt_counts, n_workers=1)

    print('\n' + '='*60)
    print('FINAL RESULTS')
    print('='*60)
    print('\nOptimised parameters:')
    for name, val in zip(PARAM_NAMES, best_params):
        print(f'  {name:<22} : {val:12.6f}')
    print(f'\nTraining-set metrics (n={len(gt_counts)}):')
    print(f'  MAPE : {best_score:8.4f} %')
    print(f'  MPE  : {final_m["MPE"]:8.2f} %')
    print(f'  MAE  : {final_m["MAE"]:8.2f} ± {final_m["STD"]:.2f} cysts')
    print(f'  RMSE : {final_m["RMSE"]:8.2f} cysts')
    print(f'  R²   : {final_m["R_squared"]:8.4f}')

    # ---- Save outputs -----------------------------------------------------
    np.savetxt(output_dir / 'BestParams.txt', best_params)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / 'FitnessHistory.txt', index=False, sep='\t')

    _plot_convergence(history, output_dir)
    _plot_predictions(final_m, gt_counts, output_dir)
    _write_report(best_params, best_score, final_m, history, output_dir, gt_counts)

    print(f'\nResults written to: {output_dir}\n')


# ---- Plotting helpers -----------------------------------------------------

def _plot_convergence(history, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(history['generation'], history['best_score'], 'b-o', ms=4, lw=2, label='Best')
    ax.plot(history['generation'], history['mean_score'], 'r--', lw=1.5, label='Mean')
    ax.set_xlabel('Generation'); ax.set_ylabel('MAPE (%)'); ax.set_title('Fitness convergence')
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.errorbar(history['generation'], history['MAE'], yerr=history['STD'],
                fmt='b-o', ms=4, lw=2, capsize=5)
    ax.set_xlabel('Generation'); ax.set_ylabel('MAE (cysts)'); ax.set_title('MAE ± STD')
    ax.grid(True)

    ax = axes[0, 2]
    ax.plot(history['generation'], history['R2'], 'g-o', ms=4, lw=2)
    ax.set_xlabel('Generation'); ax.set_ylabel('R²'); ax.set_title('R² convergence')
    ax.set_ylim([0, 1]); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(history['generation'], history['MPE'], 'm-o', ms=4, lw=2)
    ax.axhline(0, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Generation'); ax.set_ylabel('MPE (%)'); ax.set_title('Bias convergence')
    ax.grid(True)

    # Remove unused subplots
    axes[1, 1].set_visible(False)
    axes[1, 2].set_visible(False)

    fig.suptitle('GA convergence', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'GA_convergence.png', dpi=150)
    plt.close(fig)


def _plot_predictions(m, gt_counts, output_dir):
    gt   = m['ground_truth']
    pred = m['predictions']
    lo   = min(gt.min(), pred.min())
    hi   = max(gt.max(), pred.max())
    pad  = (hi - lo) * 0.1
    ax_range = [lo - pad, hi + pad]
    p = np.polyfit(gt, pred, 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(gt, pred, s=80, alpha=0.8, color='#3366CC', zorder=5)
    ax.plot(ax_range, ax_range, 'k--', lw=2, label='Identity')
    xf = np.linspace(ax_range[0], ax_range[1], 100)
    ax.plot(xf, np.polyval(p, xf), 'r-', lw=2, label='Regression')
    ax.set_xlim(ax_range); ax.set_ylim(ax_range)
    ax.set_xlabel('Ground truth (cysts)', fontweight='bold')
    ax.set_ylabel('Predicted (cysts)',    fontweight='bold')
    ax.set_title('Predictions vs ground truth (training set)', fontweight='bold')
    txt = (f'y = {p[0]:.3f}x + {p[1]:.2f}\n'
           f'MAPE: {m["MAPE"]:.2f}%\nR²: {m["R_squared"]:.4f}\n'
           f'MAE: {m["MAE"]:.2f} ± {m["STD"]:.2f}')
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            va='top', fontsize=10, bbox=dict(boxstyle='round', fc='white', ec='black'))
    ax.legend(loc='lower right'); ax.grid(True); ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_dir / 'predictions_scatter.png', dpi=150)
    fig.savefig(output_dir / 'predictions_scatter.svg')
    plt.close(fig)


def _write_report(best_params, best_score, m, history, output_dir, gt_counts):
    from datetime import datetime
    with open(output_dir / 'optimization_report.txt', 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('                      GA OPTIMISATION REPORT\n')
        f.write('=' * 80 + '\n')
        f.write(f'Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n')
        f.write(f'Training samples : {len(gt_counts)}\n')
        f.write(f'Count range      : [{gt_counts.min()} – {gt_counts.max()}] cysts\n')
        f.write(f'Mean ± SD        : {gt_counts.mean():.1f} ± {gt_counts.std():.1f} cysts\n\n')
        f.write('Optimised parameters:\n')
        for name, val in zip(PARAM_NAMES, best_params):
            f.write(f'  {name:<22} : {val:14.6f}\n')
        f.write('\nTraining-set metrics:\n')
        f.write(f'  MAPE : {best_score:8.4f} %\n')
        f.write(f'  MPE  : {m["MPE"]:8.2f} %\n')
        f.write(f'  MAE  : {m["MAE"]:8.2f} ± {m["STD"]:.2f} cysts\n')
        f.write(f'  RMSE : {m["RMSE"]:8.2f} cysts\n')
        f.write(f'  R²   : {m["R_squared"]:8.4f}\n\n')
        f.write(f'  {"Sample":>8}  {"Ground truth":>12}  {"Predicted":>12}  {"Error":>8}  {"% Error":>9}\n')
        f.write('  ' + '-' * 58 + '\n')
        for i in range(len(gt_counts)):
            err = m['predictions'][i] - gt_counts[i]
            pct = abs(err) / gt_counts[i] * 100
            f.write(f'  {i+1:>8}  {int(gt_counts[i]):>12}  {int(m["predictions"][i]):>12}  '
                    f'{int(err):>+8}  {pct:>8.2f}%\n')
        f.write('\n' + '=' * 80 + '\n')


if __name__ == '__main__':
    run_optimization()
