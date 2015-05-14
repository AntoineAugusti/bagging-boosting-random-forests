clear all;
close all;
clc;

fprintf('Loading the diabetes dataset\n');
raw = load('datasets/diabetes.mat');
ratioTrainingSet = 0.7;
[Xapp, Yapp, Xtest, Ytest] = split(raw.X, raw.Y, ratioTrainingSet);
fprintf('Keeping %f %% of the data in the training set\n', ratioTrainingSet * 100);

nbBags = 100;
nbFolds = length(Xapp);

fprintf('Running bagging with a knn classifier\n');
fprintf('%d bags with %d folds per bag\n', nbBags, nbFolds);
tic;classifiers = baggingTrain(Xapp, Yapp, 'knn', nbBags, nbFolds);toc
tic;baggingPredictions = baggingPredict(classifiers, Xtest, Ytest);toc;
err = baggingError(baggingPredictions, Ytest);
fprintf('Error made: %f %%\n', err * 100);
