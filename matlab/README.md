# MATLAB pipeline

## Requirements

- MATLAB R2020b or later
- Toolboxes: **Image Processing**, **Statistics & Machine Learning**, **Parallel Computing**, **Global Optimization**

## Data setup

Download the dataset from Zenodo and place the two folders inside `data/` at the repository root:

```
CYSTSCAN-PKD/
└── data/
    └── 2_Cyst_counting_pipeline/
        ├── Optimization set/
        │   ├── Cyst masks/          (10 .nii.gz files)
        │   └── O1_count.xlsx
        └── Evaluation set/
            ├── Cyst masks/          (10 .nii.gz files)
            └── O1_O2_counts.xlsx
```

## Files

| File | Description |
|------|-------------|
| `detectCystPeaks.m` | Core function: counts cysts in one mask given a parameter set |
| `fitness_function.m` | Evaluates a parameter set against all training masks; returns MAPE |
| `run_optimization.m` | Runs the genetic algorithm on the Optimization set |
| `run_evaluation.m` | Evaluates on the Evaluation set using GA-generated parameters |
| `run_evaluation_paper.m` | Same, using the published parameters (no prior GA run needed) |
| `sensitivity_analysis.m` | Morris screening + OAT + 2-D interaction analysis |
| `statistical_analysis.R` | Bland-Altman and regression plots (R, reads the CSV from evaluation) |

## Usage

**Step 1 – Optimise parameters:**

```matlab
run_optimization
```

Writes `results/optimization/BestParams.txt` and convergence plots.

**Step 2 – Evaluate on held-out cases:**

```matlab
run_evaluation          % uses BestParams.txt from Step 1
% or
run_evaluation_paper    % uses the published parameters, no Step 1 needed
```

Writes `results/evaluation/evaluation_results.csv` and scatter plots.

**Step 3 (optional) – Sensitivity analysis:**

```matlab
sensitivity_analysis
```

**Step 4 (optional) – Statistical plots:**

Open `statistical_analysis.R` in RStudio and run it, or from the terminal:

```bash
Rscript matlab/statistical_analysis.R
```

Requires R packages `readr`, `ggplot2`, `dplyr`:

```r
install.packages(c("readr", "ggplot2", "dplyr"))
```

Reads `results/evaluation/evaluation_results.csv` and writes Bland-Altman and regression plots to `results/evaluation/plots/`.

## Outputs

```
results/
├── optimization/
│   ├── BestParams.txt
│   ├── FitnessHistory.txt
│   ├── GA_convergence.png / .fig
│   ├── predictions_scatter.png / .fig / .svg
│   └── optimization_report.txt
├── evaluation/                    (or evaluation_paper/)
│   ├── evaluation_results.csv
│   ├── scatter.png / .svg
│   └── plots/
│       ├── bland_altman_op1_op2.png
│       ├── bland_altman_op1_auto.png
│       ├── bland_altman_op2_auto.png
│       ├── regression_op1_op2.png
│       ├── regression_op1_auto.png
│       └── regression_op2_auto.png
└── sensitivity_analysis/
    ├── morris_screening.png
    ├── oat_analysis.png
    ├── interaction_2d.png
    └── sensitivity_analysis_<timestamp>.mat
```
