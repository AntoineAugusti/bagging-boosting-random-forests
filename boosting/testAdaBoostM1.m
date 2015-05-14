clear all;
close all;
clc;

fprintf('Running the AdaBoost M1 algorithm on the synth8 dataset\n');

load 'datasets/synth8.mat';
% Keep 60% of the data for the training dataset
ratioTrainSet = 0.6;
[Xapp, Yapp, Xtest, Ytest] = split(X, Y, ratioTrainSet);
fprintf('%f %% of the data is in the training set\n', ratioTrainSet * 100);

% Run AdaBoost M1
nbIterations = 20;
fprintf('Performing %d iterations using tree classifiers\n', nbIterations);
tic
[classifiers, classifiersWeights] = adaBoostM1Train(Xapp, Yapp, nbIterations);
toc

% Get predictions on the test set
preds = adaBoostM1Predict(Xtest, length(unique(Ytest)), classifiers, classifiersWeights);

% Display the number of errors
fprintf('Percentage of errors %f %%.\n', computeError(preds, Ytest));
