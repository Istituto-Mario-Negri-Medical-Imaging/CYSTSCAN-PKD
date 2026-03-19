function peakCount = detectCystPeaks(maskPath, params)
% detectCystPeaks  Count cysts in a segmentation mask via distance-transform peak detection.
%
%   peakCount = detectCystPeaks(maskPath, params)
%
%   maskPath : full path to a NIfTI mask file (.nii or .nii.gz).
%              Voxel label 2 = cyst, all other labels are ignored.
%   params   : 1×6 numeric vector
%                params(1)  minPeakSeparationSmallCyst  – minimum Euclidean distance
%                           (voxels) between accepted peaks inside a small cyst region
%                params(2)  minPeakSeparationLargeCyst  – same, for large cyst regions
%                params(3)  largeCystVolumeThreshold    – volume (voxels³) above which
%                           a connected component is treated as a large cyst
%                params(4)  maxPeakDistanceToMerge      – large-cyst peaks closer than
%                           this distance are merged into one
%                params(5)  thresholdFraction           – adaptive threshold =
%                           mean + thresholdFraction × std of local distance values
%                params(6)  gaussianFilterSigma         – sigma for 3-D Gaussian
%                           smoothing of the distance transform
%
%   peakCount : integer, number of cysts detected in the mask.

    minPeakSeparationSmallCyst = params(1);
    minPeakSeparationLargeCyst = params(2);
    largeCystVolumeThreshold   = params(3);
    maxPeakDistanceToMerge     = params(4);
    thresholdFraction          = params(5);
    gaussianFilterSigma        = params(6);

    % Load mask and isolate cyst voxels (label == 2)
    rawMask  = niftiread(maskPath);
    cystMask = rawMask == 2;

    % Distance transform, normalised to [0, 1]
    distanceTransform = bwdist(~cystMask);
    distanceTransform = distanceTransform / max(distanceTransform(:));

    % Per-slice adaptive histogram equalisation
    for i = 1:size(distanceTransform, 3)
        distanceTransform(:,:,i) = adapthisteq(distanceTransform(:,:,i));
    end

    % Gaussian smoothing and local-maxima detection (26-connected)
    smoothedDT  = imgaussfilt3(distanceTransform, gaussianFilterSigma);
    peakMarkers = imregionalmax(smoothedDT, 26);

    % Iterate over connected cyst components
    cc       = bwconncomp(cystMask, 26);
    numCysts = 0;

    for i = 1:cc.NumObjects
        regionIdx  = cc.PixelIdxList{i};
        cystVolume = numel(regionIdx);

        if cystVolume > largeCystVolumeThreshold
            minPeakSeparation = minPeakSeparationLargeCyst;
            mergePeaks = true;
        else
            minPeakSeparation = minPeakSeparationSmallCyst;
            mergePeaks = false;
        end

        regionDT    = smoothedDT(regionIdx);
        regionPeaks = peakMarkers(regionIdx);

        % Adaptive threshold based on local distance-transform statistics
        mu                 = mean(regionDT);
        sigma              = std(regionDT);
        adaptive_threshold = mu + thresholdFraction * sigma;
        validIdx           = regionIdx(regionPeaks & (regionDT >= adaptive_threshold));

        [x, y, z]  = ind2sub(size(cystMask), validIdx);
        peakCoords = [x, y, z];

        if isempty(peakCoords)
            continue;
        end

        % Suppress peaks that are too close together
        D = pdist2(peakCoords, peakCoords, 'euclidean');
        D = D + diag(inf(size(D, 1), 1));

        uniquePeaks = true(size(peakCoords, 1), 1);
        for j = 1:size(D, 1)
            if uniquePeaks(j)
                closePeaks          = D(j,:) < minPeakSeparation;
                uniquePeaks(closePeaks) = false;
                uniquePeaks(j)      = true;
            end
        end

        % Merge peaks within large cysts
        if mergePeaks
            for j = 1:size(peakCoords, 1)
                for k = j+1:size(peakCoords, 1)
                    if norm(peakCoords(j,:) - peakCoords(k,:)) < maxPeakDistanceToMerge
                        uniquePeaks(k) = false;
                    end
                end
            end
        end

        numCysts = numCysts + sum(uniquePeaks);
    end

    peakCount = numCysts;
end
