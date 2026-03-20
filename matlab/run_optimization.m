% run_optimization.m
% Genetic algorithm optimisation of cyst-counting parameters.
%
% Reads the Optimization set from data/, runs the GA, and writes results
% to results/optimization/.
%
% Required MATLAB toolboxes:
%   Image Processing, Statistics & Machine Learning,
%   Parallel Computing, Global Optimization

clear
clc
delete(gcp('nocreate'));

%% Paths
scriptDir  = fileparts(mfilename('fullpath'));
addpath(scriptDir);                         % makes detectCystPeaks visible to parfor workers

dataDir    = fullfile(scriptDir, '..', 'data', '2_Cyst_counting_pipeline', 'Optimization set');
maskDir    = fullfile(dataDir, 'Cyst masks');
outputDir  = fullfile(scriptDir, '..', 'results', 'optimization');

if ~exist(outputDir, 'dir'), mkdir(outputDir); end

%% Load ground truth from Excel
xlsFile = fullfile(dataDir, 'O1_count.xlsx');
if ~isfile(xlsFile)
    error('Ground-truth file not found:\n  %s\nPlace the Zenodo data inside data/.', xlsFile);
end

T = readtable(xlsFile);                     % columns: Sample, "O1 count"
gt_masks = cell(height(T), 1);
gt_am    = zeros(1, height(T));

for i = 1:height(T)
    sampleID     = T{i, 1}{1};             % e.g. 'K01'
    gt_am(i)     = T{i, 2};
    gt_masks{i}  = fullfile(maskDir, sampleIDtoFilename(sampleID));
    if ~isfile(gt_masks{i})
        error('Mask not found for sample %s:\n  %s', sampleID, gt_masks{i});
    end
end

%% Summary
fprintf('\n==================================================\n');
fprintf('  GENETIC ALGORITHM OPTIMISATION\n');
fprintf('==================================================\n\n');
fprintf('Training samples : %d\n', length(gt_am));
fprintf('Ground truth     : [%d – %d] cysts  (mean %.1f ± %.1f)\n\n', ...
    min(gt_am), max(gt_am), mean(gt_am), std(gt_am));

%% Run GA
[bestParams, bestScore, history] = runGA(gt_masks, gt_am, outputDir);

%% Save outputs
save(fullfile(outputDir, 'optimization_results.mat'), ...
     'bestParams', 'bestScore', 'history', 'gt_am', 'gt_masks');

writematrix(bestParams, fullfile(outputDir, 'BestParams.txt'));

if ~isempty(history.generation)
    fid = fopen(fullfile(outputDir, 'FitnessHistory.txt'), 'w');
    fprintf(fid, 'Generation\tBestScore\tMPE\tMAE\tR2\tPopMean\n');
    fclose(fid);
    fitData = [history.generation(:), history.best_score(:), ...
               history.best_MPE(:), history.best_MAE(:), ...
               history.best_R2(:),  history.mean_score(:)];
    writematrix(fitData, fullfile(outputDir, 'FitnessHistory.txt'), ...
                'WriteMode', 'append', 'Delimiter', 'tab');
end

fprintf('\n==================================================\n');
fprintf('  OPTIMISATION COMPLETE\n');
fprintf('==================================================\n');
fprintf('Results written to: %s\n\n', outputDir);

%% -----------------------------------------------------------------------
%  Internal functions
%% -----------------------------------------------------------------------

function [bestParams, bestScore, history] = runGA(gt_masks, gt_am, outputDir)

    numParams = 6;
    lb = [1,  10,  10000, 1,  0.1, 0.01];
    ub = [15, 100, 150000, 15, 3,  1.5];

    % Temporary file for per-generation history
    histFile = fullfile(outputDir, 'ga_history_temp.mat');
    hs = struct('generation', [], 'best_score', [], 'best_MAPE', [], ...
                'best_MPE', [], 'best_MAE', [], 'best_STD', [], ...
                'best_R2', [], 'best_params', [], 'mean_score', [], ...
                'stall_gen', 0, 'best_score_ever', inf);
    save(histFile, 'hs');

    outputFcn = @(options, state, flag) gaOutputFcn(options, state, flag, ...
                                                    gt_masks, gt_am, outputDir);

    options = optimoptions('ga', ...
        'SelectionFcn',       {@selectiontournament, 3}, ...
        'MutationFcn',        {@mutationadaptfeasible, 0.95}, ...
        'EliteCount',         2, ...
        'PopulationSize',     50, ...
        'MaxGenerations',     200, ...
        'CrossoverFraction',  0.5, ...
        'MaxStallGenerations',10, ...
        'FunctionTolerance',  1e-4, ...
        'MigrationFraction',  0.15, ...
        'UseParallel',        false, ...
        'Display',            'off', ...
        'OutputFcn',          outputFcn);

    fitFcn = @(p) fitnessFunctionWrapper(p, gt_masks, gt_am);

    fprintf('%-8s | %-15s | %-10s | %-14s | %-10s | %-10s | %-8s\n', ...
            'Gen', 'Best MAPE (%)', 'MPE (%)', 'MAE ± STD', 'R²', 'Pop Mean', 'Stall');
    fprintf('%s\n', repmat('-', 1, 95));

    [bestParams, bestScore] = ga(fitFcn, numParams, [], [], [], [], lb, ub, [], options);

    % Retrieve history
    if isfile(histFile)
        load(histFile, 'hs');
        history = hs;
        delete(histFile);
        fprintf('\nHistory: %d generations recorded\n', length(history.generation));
    else
        warning('History file missing – convergence plots will be empty.');
        history = struct('generation', []);
    end

    % Final evaluation
    [~, finalMetrics] = fitness_function(bestParams, gt_masks, gt_am);

    % Console report
    fprintf('\n%s\n', repmat('=', 1, 80));
    fprintf('FINAL RESULTS\n');
    fprintf('%s\n', repmat('=', 1, 80));
    paramNames = {'minPeakSepSmall', 'minPeakSepLarge', 'largeCystVolThresh', ...
                  'maxPeakDistMerge', 'thresholdFraction', 'gaussianSigma'};
    fprintf('\nOptimised parameters:\n');
    for i = 1:6
        fprintf('  %-22s : %10.4f\n', paramNames{i}, bestParams(i));
    end
    fprintf('\nTraining-set metrics (n=%d):\n', length(gt_am));
    fprintf('  MAPE : %8.4f %%\n',  bestScore);
    fprintf('  MPE  : %8.2f %%\n',  finalMetrics.MPE);
    fprintf('  MAE  : %8.2f ± %.2f cysts\n', finalMetrics.MAE, finalMetrics.STD);
    fprintf('  RMSE : %8.2f cysts\n', finalMetrics.RMSE);
    fprintf('  R²   : %8.4f\n',     finalMetrics.R_squared);

    fprintf('\n  Sample | Ground truth | Predicted | Error\n');
    fprintf('  %s\n', repmat('-', 1, 46));
    for i = 1:length(gt_am)
        fprintf('  %6d | %12d | %9d | %+5d\n', i, ...
                finalMetrics.ground_truth(i), finalMetrics.predictions(i), ...
                finalMetrics.predictions(i) - finalMetrics.ground_truth(i));
    end

    % Plots and text report
    plotConvergence(history, outputDir);
    plotPredictions(finalMetrics, outputDir);
    writeTextReport(bestParams, bestScore, finalMetrics, history, outputDir, gt_am);
end

% ---- Fitness wrapper ----
function score = fitnessFunctionWrapper(params, gt_masks, gt_am)
    try
        if any(~isfinite(params)) || any(params <= 0)
            score = 1e6; return;
        end
        score = fitness_function(params, gt_masks, gt_am);
        if ~isfinite(score) || score < 0
            score = 1e6;
        end
    catch
        score = 1e6;
    end
end

% ---- GA output function (tracks history, injects diversity on stall) ----
function [state, options, optchanged] = gaOutputFcn(options, state, flag, ...
                                                    gt_masks, gt_am, outputDir)
    persistent lastGen;
    optchanged = false;

    if isempty(lastGen), lastGen = -1; end
    if ~strcmp(flag, 'iter') || state.Generation == lastGen, return; end

    try
        [bestScore, bestIdx] = min(state.Score);
        if ~isfinite(bestScore) || bestScore > 1e5
            lastGen = state.Generation; return;
        end

        bestP = state.Population(bestIdx, :);
        [~, m] = fitness_function(bestP, gt_masks, gt_am);

        if ~isscalar(m.MAPE) || ~isfinite(m.MAPE), lastGen = state.Generation; return; end

        validScores = state.Score(isfinite(state.Score) & state.Score < 1e5);
        meanScore   = mean(validScores);

        histFile = fullfile(outputDir, 'ga_history_temp.mat');
        load(histFile, 'hs');

        if bestScore < hs.best_score_ever - 1e-6
            hs.best_score_ever = bestScore;
            hs.stall_gen = 0;
            fprintf('%-8d | %15.4f | %10.2f | %6.2f±%-4.2f | %10.4f | %10.4f\n', ...
                    state.Generation, bestScore, m.MPE, m.MAE, m.STD, m.R_squared, meanScore);
        else
            hs.stall_gen = hs.stall_gen + 1;
            fprintf('%-8d | %15.4f | %10.2f | %6.2f±%-4.2f | %10.4f | %10.4f | %8d\n', ...
                    state.Generation, bestScore, m.MPE, m.MAE, m.STD, m.R_squared, meanScore, hs.stall_gen);
        end

        % Diversity injection after 5 stall generations
        if hs.stall_gen >= 5
            fprintf('   >>> Diversity injection: reinitialising 50%% of population <<<\n');
            lb       = [1, 10, 10000, 1, 0.1, 0.01];
            ub       = [15, 100, 150000, 15, 3, 1.5];
            popSize  = size(state.Population, 1);
            nReinit  = round(0.5 * (popSize - 2));  % keep 2 elite
            idxReinit = randperm(popSize - 2, nReinit) + 2;
            for k = idxReinit
                state.Population(k, :) = lb + rand(1, 6) .* (ub - lb);
            end
            hs.stall_gen = 0;
            optchanged   = true;
        end

        hs.generation(end+1)   = state.Generation;
        hs.best_score(end+1)   = bestScore;
        hs.best_MPE(end+1)     = m.MPE;
        hs.best_MAE(end+1)     = m.MAE;
        hs.best_STD(end+1)     = m.STD;
        hs.best_R2(end+1)      = m.R_squared;
        hs.mean_score(end+1)   = meanScore;
        if isempty(hs.best_params)
            hs.best_params = bestP(:)';
        else
            hs.best_params(end+1,:) = bestP(:)';
        end

        save(histFile, 'hs');
        lastGen = state.Generation;

    catch ME
        fprintf('Warning (generation %d): %s\n', state.Generation, ME.message);
        lastGen = state.Generation;
    end
end

% ---- Convergence plots ----
function plotConvergence(history, outputDir)
    if isempty(history.generation), return; end

    fig = figure('Position', [100, 100, 1400, 900], 'Visible', 'off');

    subplot(2,3,1)
    plot(history.generation, history.best_score, 'b-o', 'LineWidth', 2, 'MarkerSize', 5);
    hold on;
    plot(history.generation, history.mean_score, 'r--', 'LineWidth', 1.5);
    xlabel('Generation'); ylabel('MAPE (%)');
    title('Fitness convergence'); legend('Best', 'Population mean', 'Location', 'best');
    grid on;

    subplot(2,3,2)
    errorbar(history.generation, history.best_MAE, history.best_STD, ...
             'b-o', 'LineWidth', 2, 'MarkerSize', 5, 'CapSize', 8);
    xlabel('Generation'); ylabel('MAE (cysts)');
    title('MAE ± STD'); grid on;

    subplot(2,3,3)
    plot(history.generation, history.best_R2, 'g-o', 'LineWidth', 2, 'MarkerSize', 5);
    xlabel('Generation'); ylabel('R²');
    title('R² convergence'); ylim([0, 1]); grid on;

    subplot(2,3,4)
    plot(history.generation, history.best_MPE, 'm-o', 'LineWidth', 2, 'MarkerSize', 5);
    xlabel('Generation'); ylabel('MPE (%)');
    title('Bias convergence'); yline(0, 'k--', 'LineWidth', 1.5); grid on;

    subplot(2,3,[5 6])
    lb = [1, 10, 10000, 1, 0.1, 0.01];
    ub = [15, 100, 150000, 15, 3, 1.5];
    pNorm = (history.best_params - lb) ./ (ub - lb);
    pLabels = {'P1:minPeakSmall','P2:minPeakLarge','P3:volThresh', ...
               'P4:maxDistMerge','P5:threshFrac','P6:gaussSigma'};
    plot(history.generation, pNorm, '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    xlabel('Generation'); ylabel('Normalised value [0–1]');
    title('Parameter evolution');
    legend(pLabels, 'Location', 'eastoutside', 'FontSize', 9);
    grid on;

    sgtitle('GA Convergence', 'FontSize', 14, 'FontWeight', 'bold');
    saveas(fig, fullfile(outputDir, 'GA_convergence.png'));
    saveas(fig, fullfile(outputDir, 'GA_convergence.fig'));
    close(fig);
    fprintf('Convergence plot saved.\n');
end

% ---- Prediction scatter plot ----
function plotPredictions(m, outputDir)
    fig = figure('Position', [100, 100, 700, 650], 'Visible', 'off');

    gt   = m.ground_truth(:)';
    pred = m.predictions(:)';
    rng  = [min([gt, pred]), max([gt, pred])];
    margin = diff(rng) * 0.1;
    ax   = [rng(1)-margin, rng(2)+margin];

    scatter(gt, pred, 100, 'filled', 'MarkerFaceAlpha', 0.7, 'MarkerFaceColor', [0.2 0.4 0.8]);
    hold on;
    plot(ax, ax, 'k--', 'LineWidth', 2);
    p = polyfit(gt, pred, 1);
    xf = linspace(ax(1), ax(2), 100);
    plot(xf, polyval(p, xf), 'r-', 'LineWidth', 2);

    xlabel('Ground truth (cysts)', 'FontWeight', 'bold');
    ylabel('Predicted (cysts)',    'FontWeight', 'bold');
    title('Predictions vs ground truth (training set)', 'FontWeight', 'bold');

    R2 = m.R_squared; if numel(R2) > 1, R2 = R2(1); end
    annotation_str = sprintf('y = %.3fx + %.2f\nMAPE: %.2f%%\nR²: %.4f\nMAE: %.2f ± %.2f', ...
                             p(1), p(2), m.MAPE, R2, m.MAE, m.STD);
    xl = xlim; yl = ylim;
    text(xl(1)+0.05*diff(xl), yl(2)-0.05*diff(yl), annotation_str, ...
         'VerticalAlignment', 'top', 'BackgroundColor', 'w', ...
         'EdgeColor', 'k', 'FontSize', 10);

    legend({'Samples', 'Identity', 'Regression'}, 'Location', 'southeast');
    grid on; axis square; xlim(ax); ylim(ax);
    set(gca, 'FontSize', 11, 'LineWidth', 1.5); box on;

    saveas(fig, fullfile(outputDir, 'predictions_scatter.png'));
    saveas(fig, fullfile(outputDir, 'predictions_scatter.fig'));
    saveas(fig, fullfile(outputDir, 'predictions_scatter.svg'));
    close(fig);
    fprintf('Prediction scatter plot saved.\n');
end

% ---- Text report ----
function writeTextReport(bestParams, bestScore, m, history, outputDir, gt_am)
    fid = fopen(fullfile(outputDir, 'optimization_report.txt'), 'w');
    fprintf(fid, '================================================================================\n');
    fprintf(fid, '                        GA OPTIMISATION REPORT\n');
    fprintf(fid, '================================================================================\n');
    fprintf(fid, 'Date: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));

    fprintf(fid, '--- Dataset ---\n');
    fprintf(fid, 'Training samples : %d\n', length(gt_am));
    fprintf(fid, 'Count range      : [%d – %d] cysts\n', min(gt_am), max(gt_am));
    fprintf(fid, 'Mean ± SD        : %.1f ± %.1f cysts\n\n', mean(gt_am), std(gt_am));

    fprintf(fid, '--- Optimised parameters ---\n');
    pNames = {'minPeakSepSmall','minPeakSepLarge','largeCystVolThresh', ...
              'maxPeakDistMerge','thresholdFraction','gaussianSigma'};
    for i = 1:6
        fprintf(fid, '  %-22s : %12.6f\n', pNames{i}, bestParams(i));
    end

    fprintf(fid, '\n--- Training-set metrics ---\n');
    fprintf(fid, '  MAPE : %8.4f %%\n', bestScore);
    fprintf(fid, '  MPE  : %8.2f %%\n', m.MPE);
    fprintf(fid, '  MAE  : %8.2f ± %.2f cysts\n', m.MAE, m.STD);
    fprintf(fid, '  RMSE : %8.2f cysts\n', m.RMSE);
    fprintf(fid, '  R²   : %8.4f\n\n', m.R_squared(1));

    fprintf(fid, '--- Per-sample predictions ---\n');
    fprintf(fid, '  %-8s | %-12s | %-12s | %-10s | %-10s\n', ...
            'Sample', 'Ground truth', 'Predicted', 'Error', '%% Error');
    fprintf(fid, '  %s\n', repmat('-', 1, 62));
    for i = 1:length(gt_am)
        pctErr = 100 * abs(m.predictions(i) - gt_am(i)) / gt_am(i);
        fprintf(fid, '  %-8d | %12d | %12d | %+10d | %9.2f%%\n', ...
                i, gt_am(i), m.predictions(i), m.predictions(i) - gt_am(i), pctErr);
    end

    fprintf(fid, '\n================================================================================\n');
    fclose(fid);
    fprintf('Text report saved.\n');
end

% ---- Sample ID to NIfTI filename ----
function fname = sampleIDtoFilename(id)
    if id(1) == 'K'
        prefix = 'KRATS_';
    else
        prefix = 'XRATS_';
    end
    fname = sprintf('%s%03d.nii.gz', prefix, str2double(id(2:end)));
end
