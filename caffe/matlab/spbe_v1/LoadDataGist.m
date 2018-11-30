
addpath('funcs');
addpath('funcs\search');
addpath('methods');

strDir = 'E:\yanxia\dataset\SPQ\GistRemoveMean\';

strTrainFile = [strDir 'matTrain.bin'];
strQueryFile = [strDir 'matQuery.bin'];
strBaseFile = [strDir 'matBase.bin'];
strTruthFile = [strDir 'gist_groundtruth_100.bin'];

X_train = LoadMatrixBin(strTrainFile, 'double');
X_query = LoadMatrixBin(strQueryFile, 'double');
X_base = LoadMatrixBin(strBaseFile, 'double');
truth_nn = uint32(LoadMatrixBin(strTruthFile, 'uint32'));%zero-based
fprintf('load data ok:\n');
fprintf('training data: %d*%d\n', size(X_train,1), size(X_train,2));
fprintf('query data: %d*%d\n', size(X_query,1), size(X_query,2));
fprintf('base data: %d*%d\n', size(X_base,1), size(X_base,2));
fprintf('query-base knn groundtruth: %d*%d\n', size(truth_nn,1), size(truth_nn,2));