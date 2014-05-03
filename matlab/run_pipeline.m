function run_pipeline(saveBase)

% randomly selected feature points from event kit
% we don't do PCA first (work with raw features)
HOG_INPUT = '/auto/iris-00/rn/chensun/MyTest/DT_new/codeBooks/med.merged.hog.txt';
HOF_INPUT = '/auto/iris-00/rn/chensun/MyTest/DT_new/codeBooks/med.merged.hof.txt';
MBHX_INPUT = '/auto/iris-00/rn/chensun/MyTest/DT_new/codeBooks/med.merged.mbhx.txt';
MBHY_INPUT = '/auto/iris-00/rn/chensun/MyTest/DT_new/codeBooks/med.merged.mbhy.txt';

pipeline(HOG_INPUT, [saveBase 'hog_']);
pipeline(HOF_INPUT, [saveBase 'hof_']);
pipeline(MBHX_INPUT, [saveBase 'mbhx_']);
pipeline(MBHY_INPUT, [saveBase 'mbhy_']);
end

function pipeline(inputFeat, saveBase)
% K-Means first
% Then compute R(x)
% Whitening

% Let's set learning parameters here
numClusters = 64;
sliceSize = 100000;

X = load(inputFeat);
% compute K-Means codebook
[idx, C] = kmeans(X, numClusters, 'MaxIter', 300);


X = single(X');     % each column is a data point
C = single(C');     % each column is a cluster center

featDim = size(X, 1);
numSamples = size(X, 2);
rDim = featDim * numClusters;

R0 = zeros(rDim, 1, 'single');

% construct R(x)
Rsum = zeros(rDim, 1);
for i = 1:sliceSize:numSamples
    endi = min(i+sliceSize-1, numSamples);
    R = triemb_res (X(:, i:endi), C, R0);
    Rsum = Rsum + sum(R, 2);
end
R0 = Rsum / numSamples;

% compute Sigma
% TODO: Need to normalize R first?
covR = zeros(rDim);
for i = 1:sliceSize:numSamples
    endi = min(i+sliceSize-1, numSamples);
    R = triemb_res (X(:, i:endi), C, R0);
    covR = covR + R * R';
end

[eigvec, eigval] = eig(covR);
eigval = diag(eigval);
[~, idx] = sort(eigval, 'descend');
% drop featDim first components, also drop last 1000 components
idx = idx(featDim+1:end-1000);

projMat = eigvec(:, idx);
projMat = projMat * (1./sqrt(diag(eigval(idx))));

% convert back to row based
C = double(C');
projMat = double(projMat');
R0 = double(R0');
save([saveBase 'codeBook.txt'], 'C', '-ascii');
save([saveBase 'r0.txt'], 'R0', '-ascii');
save([saveBase 'projMat.txt'], 'projMat', '-ascii');

end
