% sensitivity_analysis.m
% Three-phase sensitivity analysis of the cyst-counting parameters:
%   Phase 1 – Morris screening (Campolongo et al. 2007)
%   Phase 2 – One-at-a-time (OAT) analysis, all 6 parameters ± 30 %
%   Phase 3 – 2-D parameter interaction for the top 2 parameters
%
% Requires: run_optimization.m to have been run (reads Optimization set masks).

clear
clc

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);

%% Configuration
optimalParams = [4.27269938837114, 18.2632731580751, 42144.3492985436, ...
                 14.9647743085524, 0.111299008136904, 0.918315428393029];
paramNames    = {'minPeakSepSmall', 'minPeakSepLarge', 'largeCystVolThresh', ...
                 'maxPeakDistMerge', 'thresholdFraction', 'gaussianSigma'};

dataDir  = fullfile(scriptDir, '..', 'data', 'Optimization set');
maskDir  = fullfile(dataDir, 'Cyst masks');
xlsFile  = fullfile(dataDir, 'O1_count.xlsx');
outputDir = fullfile(scriptDir, '..', 'results', 'sensitivity_analysis');
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

if ~isfile(xlsFile)
    error('Ground-truth file not found:\n  %s\nPlace the Zenodo data inside data/.', xlsFile);
end

T   = readtable(xlsFile);
n   = height(T);
gt_masks = cell(n, 1);
gt_am    = zeros(1, n);
for i = 1:n
    id          = T{i,1}{1};
    gt_am(i)    = T{i,2};
    gt_masks{i} = fullfile(maskDir, sampleIDtoFilename(id));
end

lowerBounds = optimalParams * 0.7;
upperBounds = optimalParams * 1.3;
BATCH_SIZE  = 8;

% Parallel pool
p = gcp('nocreate');
if isempty(p)
    parpool('Processes', 8);
end

%% Phase 1 – Morris screening
fprintf('\n=== Phase 1: Morris screening ===\n');
tic_morris = tic;

r     = 20;
M     = 1000;
p_dim = length(optimalParams);

fprintf('Generating %d candidate trajectories...\n', M);
candidates = cell(M, 1);
num_levels = 6;
delta      = num_levels / (2 * (num_levels - 1));
for m = 1:M
    candidates{m} = morrisTrajectory(p_dim, delta);
end

fprintf('Computing pairwise distances...\n');
distMat = zeros(M, M);
for m = 1:M
    for l = m+1:M
        distMat(m,l) = trajectoryDistance(candidates{m}, candidates{l});
        distMat(l,m) = distMat(m,l);
    end
    if mod(m, 200) == 0, fprintf('  %d / %d\n', m, M); end
end

selectedIdx = selectMaxSpread(distMat, r);

% Build parameter sets
allParams = cell(r * (p_dim + 1), 1);
for idx = 1:r
    B = candidates{selectedIdx(idx)};
    for i = 1:(p_dim + 1)
        allParams{(idx-1)*(p_dim+1) + i} = lowerBounds + B(i,:) .* (upperBounds - lowerBounds);
    end
end

fprintf('Evaluating %d parameter sets...\n', length(allParams));
allScores = evaluateInBatches(allParams, gt_masks, gt_am, BATCH_SIZE);

% Compute elementary effects
EE = zeros(r, p_dim);
for traj = 1:r
    base  = (traj-1)*(p_dim+1);
    sTraj = allScores((base+1):(base+p_dim+1));
    B     = candidates{selectedIdx(traj)};
    for step = 1:p_dim
        dNorm = B(step+1,:) - B(step,:);
        changed = find(abs(dNorm) > 1e-6);
        if ~isempty(changed)
            EE(traj, changed) = (sTraj(step+1) - sTraj(step)) / dNorm(changed);
        end
    end
end

morrisRes.mu_star = mean(abs(EE), 1)';
morrisRes.mu      = mean(EE, 1)';
morrisRes.sigma   = std(EE, 0, 1)';

fprintf('Morris screening completed in %.1f min\n', toc(tic_morris)/60);

% Morris plot
fig1 = figure('Position', [100,100,800,600], 'Visible', 'off');
scatter(morrisRes.mu_star, morrisRes.sigma, 120, 'filled', 'MarkerFaceAlpha', 0.7);
xlabel('\mu*', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('\sigma', 'FontSize', 12, 'FontWeight', 'bold');
title('Morris screening', 'FontSize', 13, 'FontWeight', 'bold');
grid on; hold on;
for i = 1:length(paramNames)
    text(morrisRes.mu_star(i)*1.04, morrisRes.sigma(i)*1.04, paramNames{i}, 'FontSize', 10);
end
xline(median(morrisRes.mu_star), '--r', 'LineWidth', 1.5);
yline(median(morrisRes.sigma),   '--b', 'LineWidth', 1.5);
saveas(fig1, fullfile(outputDir, 'morris_screening.png'));
close(fig1);

importance_threshold = median(morrisRes.mu_star);
[~, sortIdx] = sort(morrisRes.mu_star, 'descend');

fprintf('\n%-25s %10s %10s %10s\n', 'Parameter', 'mu*', 'mu', 'sigma');
fprintf('%s\n', repmat('-', 1, 60));
for i = 1:length(paramNames)
    k = sortIdx(i);
    star = '';
    if morrisRes.mu_star(k) > importance_threshold, star = ' *'; end
    fprintf('%-25s %10.4f %10.4f %10.4f%s\n', paramNames{k}, ...
            morrisRes.mu_star(k), morrisRes.mu(k), morrisRes.sigma(k), star);
end

%% Phase 2 – OAT analysis
fprintf('\n=== Phase 2: One-at-a-time analysis ===\n');
tic_oat = tic;

variations  = linspace(-0.3, 0.3, 21);
nVar        = length(variations);
oat_order   = sortIdx;

allOATParams = cell(length(oat_order) * nVar, 1);
paramMap     = zeros(length(oat_order) * nVar, 2);
idx = 1;
for i = 1:length(oat_order)
    for v = 1:nVar
        tp = optimalParams;
        tp(oat_order(i)) = optimalParams(oat_order(i)) * (1 + variations(v));
        allOATParams{idx} = tp;
        paramMap(idx, :)  = [i, v];
        idx = idx + 1;
    end
end

[allOATScores, allOATMAEs] = evaluateInBatchesFull(allOATParams, gt_masks, gt_am, BATCH_SIZE);
elapsed_oat = toc(tic_oat);

oatResults = cell(length(oat_order), 1);
for i = 1:length(oat_order)
    mask = paramMap(:,1) == i;
    oatResults{i} = struct('param', paramNames{oat_order(i)}, ...
                           'variations', variations, ...
                           'scores', allOATScores(mask), ...
                           'MAEs', allOATMAEs(mask));
end

% OAT plots
fig2 = figure('Position', [100,100,1600,900], 'Visible', 'off');
for i = 1:length(oat_order)
    subplot(2,3,i);
    d = oatResults{i};
    yyaxis left
    plot(d.variations*100, d.scores, 'o-', 'LineWidth', 2, 'MarkerSize', 7, 'Color', [0.2 0.4 0.8]);
    ylabel('MAPE (%)', 'FontSize', 10, 'FontWeight', 'bold');
    yyaxis right
    plot(d.variations*100, d.MAEs, 's--', 'LineWidth', 2, 'MarkerSize', 7, 'Color', [0.8 0.3 0.2]);
    ylabel('MAE (cysts)', 'FontSize', 10, 'FontWeight', 'bold');
    xlabel('Variation (%)', 'FontSize', 10);
    title(sprintf('%s (rank %d)', d.param, i), 'FontSize', 11, 'FontWeight', 'bold');
    xline(0, 'r--', 'LineWidth', 2);
    grid on;
end
sgtitle('One-at-a-time sensitivity analysis', 'FontSize', 14, 'FontWeight', 'bold');
saveas(fig2, fullfile(outputDir, 'oat_analysis.png'));
close(fig2);
fprintf('OAT completed in %.1f min\n', elapsed_oat/60);

%% Phase 3 – 2-D interaction
fprintf('\n=== Phase 3: 2-D parameter interaction ===\n');
tic_2d = tic;

p1_idx    = oat_order(1);
p2_idx    = oat_order(2);
grid_size = 9;
grid_vals = linspace(-0.25, 0.25, grid_size);

all2DParams = cell(grid_size^2, 1);
gridMap     = zeros(grid_size^2, 2);
idx = 1;
for i = 1:grid_size
    for j = 1:grid_size
        tp = optimalParams;
        tp(p1_idx) = optimalParams(p1_idx) * (1 + grid_vals(i));
        tp(p2_idx) = optimalParams(p2_idx) * (1 + grid_vals(j));
        all2DParams{idx} = tp;
        gridMap(idx, :)  = [i, j];
        idx = idx + 1;
    end
end

[scores2D, maes2D] = evaluateInBatchesFull(all2DParams, gt_masks, gt_am, BATCH_SIZE);

intScores = zeros(grid_size); intMAE = zeros(grid_size);
for k = 1:length(all2DParams)
    intScores(gridMap(k,1), gridMap(k,2)) = scores2D(k);
    intMAE(gridMap(k,1), gridMap(k,2))    = maes2D(k);
end

elapsed_2d = toc(tic_2d);
fprintf('2-D interaction completed in %.1f min\n', elapsed_2d/60);

fig3 = figure('Position', [100,100,1300,550], 'Visible', 'off');
subplot(1,2,1)
imagesc(grid_vals*100, grid_vals*100, intScores); colorbar; colormap(jet);
xlabel(sprintf('%s variation (%%)', paramNames{p2_idx}), 'FontWeight', 'bold');
ylabel(sprintf('%s variation (%%)', paramNames{p1_idx}), 'FontWeight', 'bold');
title('MAPE landscape (%)'); axis square;
hold on; plot(0, 0, 'wx', 'MarkerSize', 18, 'LineWidth', 3);

subplot(1,2,2)
imagesc(grid_vals*100, grid_vals*100, intMAE); colorbar; colormap(jet);
xlabel(sprintf('%s variation (%%)', paramNames{p2_idx}), 'FontWeight', 'bold');
ylabel(sprintf('%s variation (%%)', paramNames{p1_idx}), 'FontWeight', 'bold');
title('MAE landscape (cysts)'); axis square;
hold on; plot(0, 0, 'wx', 'MarkerSize', 18, 'LineWidth', 3);

sgtitle(sprintf('Parameter interaction: %s vs %s', paramNames{p1_idx}, paramNames{p2_idx}), ...
        'FontSize', 13, 'FontWeight', 'bold');
saveas(fig3, fullfile(outputDir, 'interaction_2d.png'));
close(fig3);

%% Save
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
save(fullfile(outputDir, sprintf('sensitivity_analysis_%s.mat', timestamp)), ...
     'morrisRes', 'oatResults', 'paramNames', 'optimalParams', 'sortIdx');
fprintf('\nResults saved to: %s\n\n', outputDir);

%% -----------------------------------------------------------------------
%  Helper functions
%% -----------------------------------------------------------------------

function scores = evaluateInBatches(paramSets, gt_masks, gt_am, batchSz)
    n = length(paramSets);
    scores = zeros(n, 1);
    nB = ceil(n / batchSz);
    for b = 1:nB
        i1 = (b-1)*batchSz + 1;
        i2 = min(b*batchSz, n);
        sz = i2 - i1 + 1;
        tmp = zeros(sz, 1);
        parfor i = 1:sz
            tmp(i) = fitness_function(paramSets{i1+i-1}, gt_masks, gt_am);
        end
        scores(i1:i2) = tmp;
        fprintf('  batch %d/%d done\n', b, nB);
    end
end

function [scores, maes] = evaluateInBatchesFull(paramSets, gt_masks, gt_am, batchSz)
    n = length(paramSets);
    scores = zeros(n, 1); maes = zeros(n, 1);
    nB = ceil(n / batchSz);
    for b = 1:nB
        i1 = (b-1)*batchSz + 1;
        i2 = min(b*batchSz, n);
        sz = i2 - i1 + 1;
        tmpS = zeros(sz,1); tmpM = zeros(sz,1);
        parfor i = 1:sz
            [s, m] = fitness_function(paramSets{i1+i-1}, gt_masks, gt_am);
            tmpS(i) = s;
            tmpM(i) = m.MAE;
        end
        scores(i1:i2) = tmpS;
        maes(i1:i2)   = tmpM;
        fprintf('  batch %d/%d done\n', b, nB);
    end
end

function B = morrisTrajectory(p, delta)
    B_star = tril(ones(p+1, p), -1);
    P      = eye(p); P = P(randperm(p), :);
    D_star = diag(2*randi([0,1], 1, p) - 1);
    x_star = rand(1, p);
    J      = ones(p+1, p);
    B      = (J(1,:) .* x_star + delta * ((2*B_star - J)*D_star + J)) * P;
end

function d = trajectoryDistance(t1, t2)
    k = size(t1, 2);
    d = 0;
    for i = 1:(k+1)
        for j = 1:(k+1)
            d = d + sum((t1(i,:) - t2(j,:)).^2);
        end
    end
end

function sel = selectMaxSpread(distMat, r)
    M = size(distMat, 1);
    [~, mx] = max(distMat(:));
    [a, b]  = ind2sub([M,M], mx);
    sel     = [a, b];
    rest    = setdiff(1:M, sel);
    for iter = 3:r
        scores = arrayfun(@(c) sum(distMat(c, sel).^2), rest);
        [~, best] = max(scores);
        sel  = [sel, rest(best)];
        rest = setdiff(rest, rest(best));
    end
    sel = sort(sel);
end

function fname = sampleIDtoFilename(id)
    if id(1) == 'K', prefix = 'KRATS_'; else, prefix = 'XRATS_'; end
    fname = sprintf('%s%03d.nii.gz', prefix, str2double(id(2:end)));
end
