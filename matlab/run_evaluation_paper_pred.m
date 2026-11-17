function run_evaluation_paper_pred(predDir)
% RUN_EVALUATION_PAPER_PRED  End-to-end evaluation of the cyst-counting
% algorithm on the Evaluation set, using the cyst masks predicted by the
% selected nnU-Net ensemble (n20) instead of the manual masks.
%
%   run_evaluation_paper_pred(predDir)
%
% predDir: folder holding the ten predicted masks, named as the Evaluation
% set (KRATS_005, KRATS_014, KRATS_018, KRATS_025, KRATS_038, KRATS_066,
% XRATS_018, XRATS_033, XRATS_046, XRATS_050; .nii.gz).
% For the training-set samples (K14, K18, K38, K66) use the cross-validation
% holdout predictions (the model never trained on those cases); for the
% others, full-model inference.
%
% Published parameters are used (identical to run_evaluation_paper.m).
% Results are written to results/evaluation_paper_pred/.

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);
repoRoot = fileparts(scriptDir);

%% Published parameters
params = [4.27269938837114, 18.2632731580751, 42144.3492985436, ...
          14.9647743085524, 0.111299008136904, 0.918315428393029];

fprintf('Using published parameters:\n');
fprintf('  [%.6f, %.6f, %.2f, %.6f, %.6f, %.6f]\n\n', params);

%% Predicted mask paths
sampleIDs = {'K05'; 'K14'; 'K18'; 'K25'; 'K38'; 'K66'; 'X18'; 'X33'; 'X46'; 'X50'};
n = numel(sampleIDs);
maskPaths = cell(n, 1);
for i = 1:n
    sid = sampleIDs{i};
    if sid(1) == 'K', prefix = 'KRATS_'; else, prefix = 'XRATS_'; end
    maskPaths{i} = fullfile(predDir, sprintf('%s%03d.nii.gz', prefix, str2double(sid(2:end))));
end

%% Operator counts (from O1_O2_counts.xlsx in the evaluation set)
dataDir = fullfile(repoRoot, 'data', '2_Cyst_counting_pipeline', 'Evaluation set');
xlsFile = fullfile(dataDir, 'O1_O2_counts.xlsx');
if ~isfile(xlsFile)
    error('Ground-truth file not found:\n  %s', xlsFile);
end
T = readtable(xlsFile);
samples = T{:,1};
gt_O1   = T{:,2}';
gt_O2   = T{:,3}';

%% Run counting
predicted = zeros(1, n);

fprintf('==================================================\n');
fprintf('  END-TO-END EVALUATION – PREDICTED MASKS\n');
fprintf('==================================================\n\n');

for i = 1:n
    sid      = sampleIDs{i};
    maskPath = maskPaths{i};
    if ~isfile(maskPath)
        error('Predicted mask not found for %s:\n  %s', sid, maskPath);
    end
    % Match operator counts row by sample ID
    idx = find(strcmp(samples, sid));
    if isempty(idx)
        error('Sample %s not found in O1_O2_counts.xlsx', sid);
    end
    predicted(i) = detectCystPeaks(maskPath, params);
    fprintf('  %s : pred_mask %3d  |  O1 %3d  |  O2 %3d\n', ...
            sid, predicted(i), gt_O1(idx), gt_O2(idx));
end

%% Metrics
o1_ordered = arrayfun(@(i) gt_O1(find(strcmp(samples, sampleIDs{i}))), (1:n)');
o2_ordered = arrayfun(@(i) gt_O2(find(strcmp(samples, sampleIDs{i}))), (1:n)');
mn_ordered = (o1_ordered + o2_ordered) / 2;

[mO1, ~] = computeMetrics(predicted, o1_ordered');
[mO2, ~] = computeMetrics(predicted, o2_ordered');
[mMn, ~] = computeMetrics(predicted, mn_ordered');

fprintf('\n--- Metrics ---\n');
fprintf('  %-20s  %8s  %8s  %8s\n', '',             'vs O1',   'vs O2',   'vs mean');
fprintf('  %-20s  %8.2f  %8.2f  %8.2f\n', 'MAPE (%)',    mO1.MAPE,  mO2.MAPE,  mMn.MAPE);
fprintf('  %-20s  %8.2f  %8.2f  %8.2f\n', 'MPE (%)',     mO1.MPE,   mO2.MPE,   mMn.MPE);
fprintf('  %-20s  %8.2f  %8.2f  %8.2f\n', 'MAE (cysts)', mO1.MAE,   mO2.MAE,   mMn.MAE);
fprintf('  %-20s  %8.2f  %8.2f  %8.2f\n', 'RMSE (cysts)',mO1.RMSE,  mO2.RMSE,  mMn.RMSE);
fprintf('  %-20s  %8.4f  %8.4f  %8.4f\n', 'R²',          mO1.R2,    mO2.R2,    mMn.R2);

%% Save CSV
outputDir = fullfile(repoRoot, 'results', 'evaluation_paper_pred');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

T_out = table(sampleIDs, o1_ordered, o2_ordered, predicted(:), ...
    'VariableNames', {'Sample','Operator1','Operator2','PredMask'});
csvFile = fullfile(outputDir, 'evaluation_results_pred.csv');
writetable(T_out, csvFile);
fprintf('\nResults saved to: %s\n', csvFile);
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
