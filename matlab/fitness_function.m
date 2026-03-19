function [score, metrics] = fitness_function(params, gt_masks, gt_am)
% fitness_function  Evaluate a parameter set against ground-truth cyst counts.
%
%   [score, metrics] = fitness_function(params, gt_masks, gt_am)
%
%   params   : 1×6 vector – see detectCystPeaks for parameter definitions.
%   gt_masks : N×1 cell array of full paths to NIfTI mask files.
%   gt_am    : 1×N vector of ground-truth cyst counts.
%
%   score    : Mean Absolute Percentage Error (MAPE, %) – the quantity
%              minimised by the genetic algorithm.
%   metrics  : Struct with per-sample predictions and aggregate statistics.

    numCases       = numel(gt_am);
    predictions    = zeros(numCases, 1);
    errors         = zeros(numCases, 1);
    overestimation = zeros(numCases, 1);
    underestimation= zeros(numCases, 1);
    percent_error  = zeros(numCases, 1);
    abs_pct_error  = zeros(numCases, 1);

    parfor n = 1:numCases
        pred             = detectCystPeaks(gt_masks{n}, params);
        predictions(n)   = pred;
        errors(n)        = abs(pred - gt_am(n));
        overestimation(n)  = max(0,  pred - gt_am(n));
        underestimation(n) = max(0,  gt_am(n) - pred);
        percent_error(n)   = (pred - gt_am(n)) / gt_am(n);
        abs_pct_error(n)   = abs(percent_error(n));
    end

    % ---- Metrics ----
    MAPE      = mean(abs_pct_error) * 100;
    MPE       = mean(percent_error) * 100;
    bias_penalty = max(0, abs(MPE) - 10) * 2;
    MAE       = mean(errors);
    STD_err   = std(errors);
    RMSE      = sqrt(mean(errors .^ 2));
    MAD       = median(errors);

    SS_res    = sum((gt_am(:) - predictions(:)) .^ 2);
    SS_tot    = sum((gt_am(:) - mean(gt_am(:)))   .^ 2);
    R_squared = 1 - SS_res / SS_tot;

    mean_bias = mean(predictions(:) - gt_am(:));
    ratio_over = sum(overestimation) / (sum(underestimation) + 1e-6);

    % ---- Fitness (Option A – MAPE only, recommended for publication) ----
    score = MAPE;

    % Option B – MAPE + bias penalty for |MPE| > 10 %:
    % score = MAPE + bias_penalty;

    % ---- Output struct ----
    metrics = struct();
    metrics.MAPE       = MAPE;
    metrics.MPE        = MPE;
    metrics.MAE        = MAE;
    metrics.STD        = STD_err;
    metrics.RMSE       = RMSE;
    metrics.MAD        = MAD;
    metrics.R_squared  = R_squared;
    metrics.mean_bias  = mean_bias;
    metrics.ratio_over = ratio_over;
    metrics.predictions  = predictions(:)';
    metrics.ground_truth = gt_am(:)';
    metrics.errors       = errors(:)';
    metrics.score_components = struct('primary', MAPE, 'bias_penalty', bias_penalty);
end
