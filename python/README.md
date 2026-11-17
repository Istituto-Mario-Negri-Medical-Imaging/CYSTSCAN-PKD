# Python pipeline

## Requirements

Python 3.10 or later. From the repository root, install the package with the
analysis extras:

```bash
pip install -e ".[analysis]"
# equivalent: pip install -r python/requirements.txt
```

The scripts in this folder import the `cystscan` package, so they need this
install step.

## Data setup

Same as the MATLAB pipeline: download from Zenodo and place `1_Segmentation_pipeline/` and `2_Cyst_counting_pipeline/` under `data/` at the repository root (see the main README).

## Files

| File | Description |
|------|-------------|
| `run_optimization.py` | Runs the genetic algorithm on the Optimization set (DEAP); `--seed` sets the random seed (default 42) |
| `run_evaluation.py` | Evaluates on the Evaluation set using GA-generated parameters |
| `run_evaluation_paper.py` | Same, using the published parameters (no prior optimisation needed) |
| `sensitivity_analysis.py` | Morris screening + OAT + 2-D interaction analysis (SALib) |
| `bland_altman_plots.py` | Bland-Altman and regression plots (reads the evaluation CSV) |

The shared code lives in the `cystscan` package:

| Module | Description |
|--------|-------------|
| `cystscan.counting` | `detect_cyst_peaks`: counts cysts in one mask given a parameter set; `PUBLISHED_PARAMS` |
| `cystscan._matlab` | Python versions of MATLAB's `adapthisteq` and `imgaussfilt3`, used by the counting |
| `cystscan.fitness` | Evaluates a parameter set against all training masks; returns MAPE |
| `cystscan.evaluation` | Shared logic for the two evaluation scripts |
| `cystscan.params` | Reads the published parameters from `cystscan/data/published_params.json` |

## Usage

Run all commands from the **repository root**:

```bash
# Step 1 – optimise parameters (takes time)
python python/run_optimization.py --seed 42

# Step 2 – evaluate on held-out cases
python python/run_evaluation.py           # uses BestParams.txt from Step 1
# or
python python/run_evaluation_paper.py     # uses published parameters, no Step 1 needed

# Optional – sensitivity analysis
python python/sensitivity_analysis.py

# Optional – statistical plots
python python/bland_altman_plots.py
```

## Agreement with the MATLAB implementation

### Cyst counting (`cystscan.counting` vs `matlab/detectCystPeaks.m`)

From version 1.0.0 the Python counting gives the same counts as the MATLAB
function that produced the paper's results. On the 10 masks of the Evaluation
set, with the published parameters, both give 89, 88, 119, 101, 83, 77, 243,
188, 82 and 137 cysts (K05, K14, K18, K25, K38, K66, X18, X33, X46, X50).
`tests/test_matlab_parity.py` checks this when the Zenodo data are in `data/`.

Earlier versions of the Python script differed by up to 21 cysts per sample
(X18: 264 instead of 243). The causes, all fixed in 1.0.0:

| Step | MATLAB | Python before 1.0.0 | Python 1.0.0 |
|------|--------|---------------------|--------------|
| CLAHE | `adapthisteq` defaults: output over the full [0, 1] range, applied to every slice | `skimage` `equalize_adapthist`, rescaled to each slice's range, constant slices skipped | port of `adapthisteq` (`cystscan._matlab.adapthisteq`) |
| Gaussian smoothing | `imgaussfilt3`: kernel radius `ceil(2σ)`, replicate padding | `scipy` `gaussian_filter`: radius `4σ`, reflect padding | port of `imgaussfilt3` |
| Threshold | `std` with N−1 | `std` with N | N−1 |
| Peak order | column-major (`ind2sub`) | row-major | column-major |
| Large-cyst merge | compares every pair of peaks, suppressed ones included | compared surviving peaks only | as MATLAB |

The CLAHE and Gaussian steps account for most of the difference; the merge
loop and peak order change a few counts on samples with large cysts.

### Other pipeline components

| Aspect | MATLAB | Python |
|--------|--------|--------|
| Genetic algorithm | MATLAB `ga()` with adaptive feasible mutation/crossover | DEAP with `mutPolynomialBounded` + `cxBlend`: same bounds and population size, different internal operators |
| Morris screening | Manual trajectory generation (Campolongo et al. 2007) | `SALib.sample.morris` + `SALib.analyze.morris`: same method |
| Statistical plots | ggplot2 (R) | matplotlib |
| Parallelism | `parfor` (MATLAB Parallel Computing Toolbox) | `concurrent.futures.ProcessPoolExecutor` |

Because of the GA operators, parameters optimised in Python will not be
identical to the MATLAB ones; they should converge to similar values. The
paper's parameters come from the MATLAB optimisation.

## Outputs

Identical structure to the MATLAB pipeline under `results/`.
