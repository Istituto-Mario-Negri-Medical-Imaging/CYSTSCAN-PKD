# Python pipeline

## Requirements

Python 3.10 or later. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data setup

Same as the MATLAB pipeline: download from Zenodo and place the two folders under `data/` at the repository root.

## Files

| File | Description |
|------|-------------|
| `detect_cyst_peaks.py` | Core function: counts cysts in one mask given a parameter set |
| `fitness_function.py` | Evaluates a parameter set against all training masks; returns MAPE |
| `run_optimization.py` | Runs the genetic algorithm on the Optimization set (uses DEAP) |
| `run_evaluation.py` | Evaluates on the Evaluation set using GA-generated parameters |
| `run_evaluation_paper.py` | Same, using the published parameters (no prior optimisation needed) |
| `sensitivity_analysis.py` | Morris screening + OAT + 2-D interaction analysis (uses SALib) |
| `bland_altman_plots.py` | Bland-Altman and regression plots (reads the evaluation CSV) |
| `_evaluation_core.py` | Shared logic for the two evaluation scripts (not called directly) |

> `detect_cyst_peaks` and `PUBLISHED_PARAMS` from `run_evaluation_paper.py` are
> also imported by `segmentation/count_cysts.py` when counting cysts on new
> inference outputs.

## Usage

```bash
# Step 1 – optimise parameters (takes time)
python run_optimization.py

# Step 2 – evaluate on held-out cases
python run_evaluation.py           # uses BestParams.txt from Step 1
# or
python run_evaluation_paper.py     # uses published parameters, no Step 1 needed

# Optional – sensitivity analysis
python sensitivity_analysis.py

# Optional – statistical plots
python bland_altman_plots.py
```

## Differences from the MATLAB implementation

### Cyst counting algorithm (`detect_cyst_peaks.py` vs `detectCystPeaks.m`)

| Aspect | MATLAB | Python |
|--------|--------|--------|
| Distance transform | `bwdist(~cystMask)` | `edt.edt()` (if installed) or `scipy.ndimage.distance_transform_edt` — functionally equivalent |
| CLAHE | `adapthisteq(slice)` per Z-slice, default `Range='original'` (output mapped back to input range) | `skimage.exposure.equalize_adapthist(slice)` per Z-slice, then explicitly rescaled to input range to match MATLAB behaviour |
| Gaussian smoothing | `imgaussfilt3` | `scipy.ndimage.gaussian_filter` — equivalent |
| Local maxima | `imregionalmax(smoothedDT, 26)` | `skimage.morphology.local_maxima(smoothed, connectivity=3)` — 26-connected, equivalent |
| Connected components | `bwconncomp(cystMask, 26)` | `scipy.ndimage.label` with full 3×3×3 structure — 26-connected, equivalent |
| **Peak merge loop** | Iterates over **all** peaks (including those suppressed in the prior step); a suppressed peak can still suppress another via the merge criterion | Iterates only over **surviving** peaks after the suppression step; suppressed peaks do not participate in merging |

The peak-merge-loop difference is the primary source of count discrepancies on samples with many large cysts.
MATLAB's behaviour is more aggressive (lower final counts on such samples).

**Evaluation set: per-sample counts (published parameters)**

| Sample | MATLAB | Python | Δ |
|--------|--------|--------|---|
| K05 | 89 | 89 | 0 |
| K14 | 88 | 88 | 0 |
| K18 | 119 | 121 | +2 |
| K25 | 101 | 102 | +1 |
| K38 | 83 | 81 | −2 |
| K66 | 77 | 81 | +4 |
| X18 | 243 | 264 | **+21** |
| X33 | 188 | 182 | −6 |
| X46 | 82 | 83 | +1 |
| X50 | 137 | 142 | +5 |

X18 shows the largest discrepancy (+21, ~8.6%) because it contains the highest number of large-cyst components where the merge step is active.

**Per-sample wall-clock time (Python, published parameters, 10-case evaluation set)**

| Sample | Time (s) |
|--------|----------|
| K05 | 15 |
| K14 | 13 |
| K18 | 57 |
| K25 | 12 |
| K38 | 15 |
| K66 | 14 |
| X18 | 60 |
| X33 | 58 |
| X46 | 52 |
| X50 | 58 |
| **Mean** | **35** |
| **Total** | **354** |

XRATS cases are slower than KRATS cases due to their larger volume and higher cyst density. To compare against MATLAB, add `tic/toc` around each `detectCystPeaks` call inside `matlab/run_evaluation_paper.m`.

### Other pipeline components

| Aspect | MATLAB | Python |
|--------|--------|--------|
| Genetic algorithm | MATLAB `ga()` with adaptive feasible mutation/crossover | DEAP with `mutPolynomialBounded` + `cxBlend` — same bounds and population size, different internal operators |
| Morris screening | Manual trajectory generation (Campolongo et al. 2007) | `SALib.sample.morris` + `SALib.analyze.morris` — methodologically equivalent |
| Statistical plots | ggplot2 (R) | matplotlib |
| Parallelism | `parfor` (MATLAB Parallel Computing Toolbox) | `concurrent.futures.ProcessPoolExecutor` |

Because of these differences – particularly in the GA operators – optimised parameters will not be bit-for-bit identical to the MATLAB results, but should converge to similar values.

## Outputs

Identical structure to the MATLAB pipeline under `results/`.
