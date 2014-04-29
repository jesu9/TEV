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
% Compute covariance matrix Sigma of R(x)
% Compute Sigma^(-1/2)
% Compute PCA using \phi

% Need to store: K-Means, Sigma^(-1/2), PCA

% Let's set learning parameters here
numClusters = 64;
sampledPts = 100000;

X = load(inputFeat);

featDim = size(X, 2);

% compute K-Means codebook
[idx, C] = kmeans(X, numClusters);
%C = load([saveBase 'kmeans.txt']);

X = X(randsample(size(X, 1), sampledPts), :);    % sample points

% construct R(x)
R = repmat(X, 1, numClusters);
CC = reshape(C', 1, size(R, 2));

R = R - repmat(CC, size(R, 1), 1);

% normalize for each codeword
for i = 1:numClusters
    % use nnet toolbox
    R(:, (i-1)*featDim+1:i*featDim) = normr(R(:, (i-1)*featDim+1:i*featDim));
end

% compute R0
R0 = mean(R, 1);

% compute Sigma
Sigma = cov(R);
% compute Sigma^-0.5
SigmaInv = Sigma^(-0.5);

% compute Phi
Phi = (R - repmat(R0, size(R, 1), 1)) * SigmaInv';
% PCA
Eig = pca(Phi);
% Drop first featDim eigenvectors
Eig = Eig(:, featDim+1:end);

% Combine PCA matrix and SigmaInv
projMat = Eig'*SigmaInv;

save([saveBase 'kmeans.txt'], 'C', '-ascii');
save([saveBase 'proj.txt'], 'projMat', '-ascii');
save([saveBase 'r0.txt'], 'R0', '-ascii');

end
