# Paper outputs and the scripts that produce them

Article: Scientific Reports (2026), doi 10.1038/s41598-026-61818-7. "Fig." and
"Table" refer to the main text, "Supplementary" to the Supplementary
Information. Numbers checked against the published article.

The paper's analysis was run with the MATLAB scripts (plots in R). The Python
scripts write the same files under the same names.

| Paper item | Content | MATLAB / R | Python | Output under `results/` |
|------------|---------|------------|--------|--------------------------|
| Not shown in the article | GA convergence (fitness, MAE, R², bias per generation) | `matlab/run_optimization.m` | `python/run_optimization.py` | `optimization/GA_convergence.png` |
| Results text; optimised values also in Supplementary Table S4 | Optimised parameters and training-set metrics | `matlab/run_optimization.m` | `python/run_optimization.py` | `optimization/BestParams.txt`, `optimization/optimization_report.txt` |
| Not shown in the article | Predicted vs manual counts, Optimization set | `matlab/run_optimization.m` | `python/run_optimization.py` | `optimization/predictions_scatter.png` |
| Supplementary Table S5 | Automatic and manual counts, Evaluation set, published parameters | `matlab/run_evaluation_paper.m` | `python/run_evaluation_paper.py` | `evaluation_paper/evaluation_results.csv`, `evaluation_paper/scatter.png` |
| Not shown in the article | Counts on the masks predicted by the segmentation models | `matlab/run_evaluation_paper_pred.m` | none | CSV in the folder given as argument |
| Fig. 5 | Bland-Altman and regression plots (operators vs automatic, Evaluation set) | `matlab/statistical_analysis.R` | `python/bland_altman_plots.py` | `evaluation*/plots/bland_altman_*.png`, `regression_*.png` |
| Supplementary Fig. S4 and Table S4 | Morris screening of the six parameters | `matlab/sensitivity_analysis.m` | `python/sensitivity_analysis.py` | `sensitivity_analysis/morris_screening.png` |
| Supplementary Fig. S5 | One-at-a-time parameter sweeps (±30%) | `matlab/sensitivity_analysis.m` | `python/sensitivity_analysis.py` | `sensitivity_analysis/oat_analysis.png` |
| Not shown in the article | Two-parameter interaction maps | `matlab/sensitivity_analysis.m` | `python/sensitivity_analysis.py` | `sensitivity_analysis/interaction_2d.png` |
| Masks behind Tables 2-3, Figs. 3-4 and Supplementary Fig. S1 | Kidney and cyst segmentation masks (n5 to n20 ensembles) | nnU-Net v2 inference: `cystscan segment` | same | masks in the output folder |

Not in this repository: the code that computed the segmentation metrics of
Tables 2-3 (Dice, IoU, HD99) and the volume Bland-Altman plots of Figs. 3-4,
the Sobel-channel comparison (Supplementary Figs. S2-S3, Table S3) and the
dataset tables (Table 1, Supplementary Tables S1-S2, S6).
