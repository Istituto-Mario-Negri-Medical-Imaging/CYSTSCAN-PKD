# Paper outputs and the scripts that produce them

Draft: the mapping from output to script is read from the code; the paper's
figure and table numbers still have to be checked against the article
(Sci Rep 2026, doi 10.1038/s41598-026-61818-7) and filled in.

The paper's analysis was run with the MATLAB scripts (plots in R). The Python
scripts write the same files under the same names.

| Paper item | Content | MATLAB / R | Python | Output under `results/` |
|------------|---------|------------|--------|--------------------------|
| _to fill_ | GA convergence (fitness, MAE, R², bias per generation) | `matlab/run_optimization.m` | `python/run_optimization.py` | `optimization/GA_convergence.png` |
| _to fill_ | Optimised parameters and training-set metrics | `matlab/run_optimization.m` | `python/run_optimization.py` | `optimization/BestParams.txt`, `optimization/optimization_report.txt` |
| _to fill_ | Predicted vs manual counts, Optimization set | `matlab/run_optimization.m` | `python/run_optimization.py` | `optimization/predictions_scatter.png` |
| _to fill_ (Supplementary Table S5: raw counts) | Automatic and manual counts, Evaluation set, published parameters | `matlab/run_evaluation_paper.m` | `python/run_evaluation_paper.py` | `evaluation_paper/evaluation_results.csv`, `evaluation_paper/scatter.png` |
| _to fill_ | Counts on the masks predicted by the segmentation models | `matlab/run_evaluation_paper_pred.m` | none | CSV in the folder given as argument |
| _to fill_ | Bland-Altman and regression plots (operators vs automatic) | `matlab/statistical_analysis.R` | `python/bland_altman_plots.py` | `evaluation*/plots/bland_altman_*.png`, `regression_*.png` |
| _to fill_ (Supplementary Table S4: Morris screening) | Morris screening of the six parameters | `matlab/sensitivity_analysis.m` | `python/sensitivity_analysis.py` | `sensitivity_analysis/morris_screening.png` |
| _to fill_ | One-at-a-time parameter sweeps | `matlab/sensitivity_analysis.m` | `python/sensitivity_analysis.py` | `sensitivity_analysis/oat_analysis.png` |
| _to fill_ | Two-parameter interaction maps | `matlab/sensitivity_analysis.m` | `python/sensitivity_analysis.py` | `sensitivity_analysis/interaction_2d.png` |
| _to fill_ | Segmentation masks (Dice and counts per model) | nnU-Net v2 inference: `cystscan segment` | same | masks in the output folder |
