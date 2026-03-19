% run_evaluation.m
% Batch evaluation of the cyst-counting algorithm on the Evaluation set,
% using the parameters produced by run_optimization.m.
%
% Run run_optimization.m first to generate results/optimization/BestParams.txt.
% To evaluate with the published parameters instead, use run_evaluation_paper.m.

clear
clc

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);

%% Load optimised parameters
paramsFile = fullfile(scriptDir, '..', 'results', 'optimization', 'BestParams.txt');
if ~isfile(paramsFile)
    error(['BestParams.txt not found.\n' ...
           'Run run_optimization.m first, or use run_evaluation_paper.m\n' ...
           'to evaluate with the published parameters.\n' ...
           'Expected file: %s'], paramsFile);
end
params = readmatrix(paramsFile)';
fprintf('Parameters loaded from: %s\n', paramsFile);

%% Run evaluation
runEvaluation(params, scriptDir, fullfile(scriptDir, '..', 'results', 'evaluation'));

%% -----------------------------------------------------------------------
function runEvaluation(params, scriptDir, outputDir)

    if ~exist(outputDir, 'dir'), mkdir(outputDir); end

    dataDir = fullfile(scriptDir, '..', 'data', 'Evaluation set');
    maskDir = fullfile(dataDir, 'Cyst masks');
    xlsFile = fullfile(dataDir, 'O1_O2_counts.xlsx');

    if ~isfile(xlsFile)
        error('Ground-truth file not found:\n  %s\nPlace the Zenodo data inside data/.', xlsFile);
    end

    T = readtable(xlsFile);   % columns: Sample, O1 count, O2 count

    n        = height(T);
    samples  = T{:, 1};
    gt_O1    = T{:, 2}';
    gt_O2    = T{:, 3}';
    gt_mean  = (gt_O1 + gt_O2) / 2;
    predicted = zeros(1, n);

    fprintf('\n==================================================\n');
    fprintf('  EVALUATION\n');
    fprintf('==================================================\n\n');
    fprintf('Parameters: [%.3f, %.3f, %.0f, %.3f, %.4f, %.4f]\n\n', params);
    fprintf('Processing %d cases...\n\n', n);

    for i = 1:n
        sampleID   = samples{i};
        maskPath   = fullfile(maskDir, sampleIDtoFilename(sampleID));
        if ~isfile(maskPath)
            error('Mask not found for sample %s:\n  %s', sampleID, maskPath);
        end
        predicted(i) = detectCystPeaks(maskPath, params);
        fprintf('  %s : predicted %3d  |  O1 %3d  |  O2 %3d\n', ...
                sampleID, predicted(i), gt_O1(i), gt_O2(i));
    end

    %% Metrics
    [mO1, pO1]  = computeMetrics(predicted, gt_O1);
    [mO2, pO2]  = computeMetrics(predicted, gt_O2);
    [mMn, pMn]  = computeMetrics(predicted, gt_mean);

    fprintf('\n--- Metrics ---\n');
    fprintf('  %-20s  %8s  %8s  %8s\n', '',          'vs O1',  'vs O2',  'vs mean');
    fprintf('  %-20s  %8.2f  %8.2f  %8.2f\n', 'MAPE (%)',  mO1.MAPE, mO2.MAPE, mMn.MAPE);
    fprintf('  %-20s  %8.2f  %8.2f  %8.2f\n', 'MPE (%)',   mO1.MPE,  mO2.MPE,  mMn.MPE);
    fprintf('  %-20s  %8.2f  %8.2f  %8.2f\n', 'MAE (cysts)',mO1.MAE, mO2.MAE,  mMn.MAE);
    fprintf('  %-20s  %8.2f  %8.2f  %8.2f\n', 'RMSE (cysts)',mO1.RMSE,mO2.RMSE,mMn.RMSE);
    fprintf('  %-20s  %8.4f  %8.4f  %8.4f\n', 'R²',        mO1.R2,   mO2.R2,   mMn.R2);

    %% Save CSV (column names match statistical_analysis.R expectations)
    csvFile = fullfile(outputDir, 'evaluation_results.csv');
    T_out   = table(samples, gt_O1(:), gt_O2(:), predicted(:), ...
                    'VariableNames', {'Sample', 'Operator 1', 'Operator 2', 'Auto'});
    writetable(T_out, csvFile);
    fprintf('\nResults saved to: %s\n', csvFile);

    %% Scatter plot
    fig = figure('Position', [100, 100, 900, 400], 'Visible', 'off');

    subplot(1,2,1)
    scatter(gt_O1, predicted, 80, 'filled', 'MarkerFaceAlpha', 0.8);
    hold on; plotIdentityAndReg(gt_O1, predicted, pO1);
    xlabel('Operator 1 (cysts)'); ylabel('Predicted (cysts)');
    title(sprintf('vs Operator 1  (R²=%.3f, MAPE=%.1f%%)', mO1.R2, mO1.MAPE));
    grid on; axis square;

    subplot(1,2,2)
    scatter(gt_O2, predicted, 80, 'filled', 'MarkerFaceAlpha', 0.8, 'MarkerFaceColor', [0.8 0.3 0.2]);
    hold on; plotIdentityAndReg(gt_O2, predicted, pO2);
    xlabel('Operator 2 (cysts)'); ylabel('Predicted (cysts)');
    title(sprintf('vs Operator 2  (R²=%.3f, MAPE=%.1f%%)', mO2.R2, mO2.MAPE));
    grid on; axis square;

    sgtitle('Evaluation set – predicted vs manual count', 'FontWeight', 'bold');
    saveas(fig, fullfile(outputDir, 'scatter.png'));
    saveas(fig, fullfile(outputDir, 'scatter.svg'));
    close(fig);
    fprintf('Scatter plot saved.\n\n');
end

%% Helpers

function [m, p] = computeMetrics(pred, gt)
    err    = abs(pred - gt);
    pctErr = err ./ gt;
    m.MAPE = mean(pctErr) * 100;
    m.MPE  = mean((pred - gt) ./ gt) * 100;
    m.MAE  = mean(err);
    m.RMSE = sqrt(mean(err .^ 2));
    SS_res = sum((gt(:) - pred(:)) .^ 2);
    SS_tot = sum((gt(:) - mean(gt(:))) .^ 2);
    m.R2   = 1 - SS_res / SS_tot;
    p = polyfit(gt(:)', pred(:)', 1);
end

function plotIdentityAndReg(gt, pred, p)
    allVals = [gt(:); pred(:)];
    rng = [min(allVals), max(allVals)];
    margin = diff(rng) * 0.1;
    ax = [rng(1)-margin, rng(2)+margin];
    plot(ax, ax, 'k--', 'LineWidth', 1.5);
    xf = linspace(ax(1), ax(2), 100);
    plot(xf, polyval(p, xf), 'r-', 'LineWidth', 1.5);
    xlim(ax); ylim(ax);
    legend({'Samples','Identity','Regression'}, 'Location', 'southeast', 'FontSize', 8);
end

function fname = sampleIDtoFilename(id)
    if id(1) == 'K', prefix = 'KRATS_'; else, prefix = 'XRATS_'; end
    fname = sprintf('%s%03d.nii.gz', prefix, str2double(id(2:end)));
end
