% function [classifiers, classifiersWeights] = adaBoostM1Train(X, Y, nbIter)
% 	Train a training dataset with the AdaBoost M1 algorithm using a tree classifier
% 	Input:
% 	- X: N*M data matrix where N is the number of examples and M the number of features
% 	- Y: N*1 matrix corresponding to the label of each element in X
% 	- nbIter: the number of iterations to run, telling the number of classifiers
% 	to train
% 	Output:
% 	- classifiers: nbIter*1 cell array of classifiers
% 	- classifiersWeights: nbIter*1 weight matrix for each classifier (weights in [0..1])
function [classifiers, classifiersWeights] = adaBoostM1Train(X, Y, nbIter)
	[n, p] = size(X);
	dataWeights = ones(n, 1) / n;
	% Find the number of classes
	C = length(unique(Y));

	classifiers = cell(nbIter, 1);
	classifiersWeights = [];
	dataset = prdataset(X, Y);
	for i = 1:nbIter
		% Generate the weighted dataset
		weightedDataset = gendatw(dataset, dataWeights);
		% Train the classifier with the weighted dataset
		classifier = treec(weightedDataset);

		% Get true labels
		labelsTrue = Y;
		% Compute a classification dataset
		D = dataset * classifier;
		% Get predicted labels
		labelsPred = D * labeld;
		errorWeighted = sum((labelsTrue ~= labelsPred)' * dataWeights);

		% Compute the weight of the current classifier
		classifierWeight = log((C - 1) * (1 - errorWeighted) / errorWeighted);

		% Recompute data weights
		for j = 1:n
			dataWeights(j) = dataWeights(j) * exp(classifierWeight * (labelsTrue(j) ~= labelsPred(j)));
		end

		% Renormalize data weights
		totalWeights = sum(dataWeights);
		dataWeights = dataWeights / totalWeights;

		% Remember the current classifier and its weight
		classifiers{i} =  classifier;
		classifiersWeights = [classifiersWeights; classifierWeight];
	end
end
