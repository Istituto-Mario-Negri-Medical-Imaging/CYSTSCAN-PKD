# statistical_analysis.R
# Bland-Altman and linear regression plots for the evaluation results.
#
# Input : results/evaluation/evaluation_results.csv
#         (auto-falls back to results/evaluation_paper/ if not found)
# Output: results/<evaluation or evaluation_paper>/plots/
#
# Required packages: readr, ggplot2, dplyr
# Install with: install.packages(c("readr", "ggplot2", "dplyr"))
#
# Run from RStudio (the script sets its working directory automatically),
# or from the terminal: Rscript matlab/statistical_analysis.R

library(readr)
library(ggplot2)
library(dplyr)

# ---- Paths ----------------------------------------------------------------

# Locate the repository root relative to this script
script_dir <- tryCatch(
  dirname(rstudioapi::getSourceEditorContext()$path),   # RStudio
  error = function(e) {
    args <- commandArgs(trailingOnly = FALSE)
    script_flag <- grep("^--file=", args, value = TRUE)
    if (length(script_flag) > 0) dirname(sub("--file=", "", script_flag))  # Rscript
    else getwd()
  }
)

repo_root <- normalizePath(file.path(script_dir, ".."))
# Auto-detect: prefer results/evaluation/, fall back to results/evaluation_paper/
results_subdir <- "evaluation"
if (!file.exists(file.path(repo_root, "results", "evaluation", "evaluation_results.csv"))) {
  results_subdir <- "evaluation_paper"
}
csv_file  <- file.path(repo_root, "results", results_subdir, "evaluation_results.csv")
plot_dir  <- file.path(repo_root, "results", results_subdir, "plots")

dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Load data ------------------------------------------------------------

if (!file.exists(csv_file)) {
  stop(paste0(
    "Results file not found:\n  ", csv_file,
    "\nRun run_evaluation.m (or run_evaluation_paper.m) first."
  ))
}

data <- read_csv(csv_file, show_col_types = FALSE)

# ---- Bland-Altman helper --------------------------------------------------

bland_altman_plot <- function(ref, test, ref_label, test_label, out_path) {
  diff_pct  <- (ref - test) / test * 100
  avg       <- (ref + test) / 2
  bias      <- mean(diff_pct)
  sd_d      <- sd(diff_pct)
  loa_upper <- bias + 1.96 * sd_d
  loa_lower <- bias - 1.96 * sd_d

  df <- data.frame(avg = avg, diff = diff_pct)

  y_range <- max(diff_pct) - min(diff_pct)

  p <- ggplot(df, aes(x = avg, y = diff)) +
    geom_point(shape = 21, color = "black", fill = "grey", size = 3, stroke = 1) +
    geom_hline(yintercept = 0,         color = "black", linewidth = 1.2) +
    geom_hline(yintercept = loa_upper, color = "blue", linetype = "dashed", linewidth = 1.2) +
    geom_hline(yintercept = bias,      color = "red",  linetype = "dashed", linewidth = 1.2) +
    geom_hline(yintercept = loa_lower, color = "blue", linetype = "dashed", linewidth = 1.2) +
    annotate("text", x = max(df$avg) * 0.93, y = loa_upper - y_range * 0.08,
             label = paste("+2\u00d7SD =", signif(loa_upper, 3)), color = "blue",
             size = 5, fontface = "bold") +
    annotate("text", x = max(df$avg) * 0.93, y = bias + y_range * 0.08,
             label = paste("Bias =", signif(bias, 3)), color = "red",
             size = 5, fontface = "bold") +
    annotate("text", x = max(df$avg) * 0.93, y = loa_lower - y_range * 0.08,
             label = paste("-2\u00d7SD =", signif(loa_lower, 3)), color = "blue",
             size = 5, fontface = "bold") +
    ggtitle(paste(ref_label, "vs", test_label)) +
    xlab("Average number of cysts") +
    ylab("Difference (%)") +
    coord_cartesian(ylim = c(-30, 30)) +
    theme(
      plot.title        = element_text(size = 18, face = "bold", hjust = 0.5),
      axis.title        = element_text(size = 14, face = "bold"),
      panel.grid.major  = element_line(color = "grey90"),
      panel.background  = element_blank(),
      plot.margin       = margin(10, 20, 10, 10)
    )

  ggsave(out_path, plot = p, width = 6, height = 4, dpi = 300)

  cat(sprintf("\n%s vs %s:\n  Bias: %s %%\n  SD:   %s %%\n  LoA:  %s %% to %s %%\n",
              ref_label, test_label,
              signif(bias, 3), signif(sd_d, 3),
              signif(loa_lower, 3), signif(loa_upper, 3)))
}

# ---- Regression helper ----------------------------------------------------

regression_plot <- function(x_vals, y_vals, x_label, y_label, out_path) {
  df    <- data.frame(x = x_vals, y = y_vals) |> filter(complete.cases(x, y))
  model <- lm(y ~ x, data = df)
  sm    <- summary(model)
  b0    <- round(coef(model)[1], 2)
  b1    <- round(coef(model)[2], 2)
  R2    <- round(sm$r.squared, 3)
  pval  <- sm$coefficients[2, 4]
  p_label <- if (pval < 0.001) "P < 0.001" else paste("P =", round(pval, 3))

  eq_label <- paste0("y = ", b1, "x + ", b0, "\nR\u00b2 = ", R2, "\n", p_label)

  xr <- range(df$x); yr <- range(df$y)
  lx <- xr[1] + 0.05 * diff(xr)
  ly <- yr[2] - 0.05 * diff(yr)

  p <- ggplot(df, aes(x = x, y = y)) +
    geom_point(size = 3, alpha = 0.7, color = "#009E73") +
    geom_smooth(method = "lm", se = TRUE, color = "black", linewidth = 1, alpha = 0.2) +
    annotate("label", x = lx, y = ly, label = eq_label,
             hjust = 0, vjust = 1, size = 5, fill = "white", color = "black",
             fontface = "bold", label.size = 0.5, alpha = 1) +
    ggtitle(paste(x_label, "vs", y_label)) +
    xlab(paste(x_label, "(number of cysts)")) +
    ylab(paste(y_label, "(number of cysts)")) +
    theme(
      plot.title        = element_text(size = 18, face = "bold", hjust = 0.5),
      axis.title        = element_text(size = 14, face = "bold"),
      axis.text         = element_text(size = 11),
      panel.grid.major  = element_line(color = "grey90"),
      panel.grid.minor  = element_blank(),
      panel.background  = element_blank(),
      plot.margin       = margin(10, 20, 10, 10)
    )

  ggsave(out_path, plot = p, width = 6, height = 4, dpi = 300, bg = "white")
}

# ---- Bland-Altman plots ---------------------------------------------------

cat("\n=== Bland-Altman plots ===\n")

bland_altman_plot(
  data$`Operator 1`, data$`Operator 2`,
  "Operator 1", "Operator 2",
  file.path(plot_dir, "bland_altman_op1_op2.png")
)

bland_altman_plot(
  data$`Operator 1`, data$Auto,
  "Operator 1", "Auto",
  file.path(plot_dir, "bland_altman_op1_auto.png")
)

bland_altman_plot(
  data$`Operator 2`, data$Auto,
  "Operator 2", "Auto",
  file.path(plot_dir, "bland_altman_op2_auto.png")
)

# ---- Regression plots -----------------------------------------------------

cat("\n=== Regression plots ===\n")

regression_plot(
  data$`Operator 1`, data$`Operator 2`,
  "Operator 1", "Operator 2",
  file.path(plot_dir, "regression_op1_op2.png")
)

regression_plot(
  data$`Operator 1`, data$Auto,
  "Operator 1", "Auto",
  file.path(plot_dir, "regression_op1_auto.png")
)

regression_plot(
  data$`Operator 2`, data$Auto,
  "Operator 2", "Auto",
  file.path(plot_dir, "regression_op2_auto.png")
)

cat(sprintf("\nAll plots saved to: %s\n", plot_dir))
