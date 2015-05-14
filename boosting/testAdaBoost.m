clear all;
close all;
clc;

fprintf('Running the AdaBoost algorithm on the diabetes dataset\n');

load 'datasets/diabetes.mat';
% Transform a vector of 1s and 0s to 1s and -1s
Y = Y * 2 - 1;

% Keep 60% of the data for the training dataset
ratioTrainSet = 0.6;
[Xapp, Yapp, Xtest, Ytest] = split(X, Y, ratioTrainSet);
fprintf('%f %% of the data is in the training set\n', ratioTrainSet * 100);

% Run AdaBoost
nbIterations = 1000;
fprintf('Performing %d iterations using decision stumps\n', nbIterations);
tic
[classifiers, classifiersWeights] = adaBoostTrain(Xapp, Yapp, nbIterations);
toc
% Get predictions on the test set
preds = adaBoostPredict(Xtest, classifiers, classifiersWeights);
% Display the number of errors
fprintf('Percentage of errors %f %%.\n', computeError(preds, Ytest));
