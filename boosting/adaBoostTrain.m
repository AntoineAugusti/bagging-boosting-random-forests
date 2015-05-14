% function [classifiers, classifiersWeights] = adaBoostTrain(X, Y, nbIter)
% 	Train a training dataset with the AdaBoost algorithm using decision stumps
% 	Input:
% 	- X: N*M data matrix where N is the number of examples and M the number of features
% 	- Y: N*1 label matrix (label -1 or 1)
% 	- nbIter: the number of iterations to run, telling the number of classifiers
% 	to train
% 	Output:
% 	- classifiers: nbIter*1 cell array of classifiers
% 	- classifiersWeights: nbIter*1 weight matrix for each classifier (weights in [0..1])
function [classifiers, classifiersWeights] = adaBoostTrain(X, Y, nbIter)
	[n, p] = size(X);
	dataWeights = ones(n, 1) / n;

	classifiers = [];
	classifiersWeights = [];
	for i = 1:nbIter
		classifier = decisionStumpTrain(X, Y, dataWeights);

		% Get predicted labels
		labelsPred = decisionStumpVal(classifier, X);
		errorWeighted = sum((Y ~= labelsPred)' * dataWeights);

		% Compute the weight of the current classifier
		classifierWeight = 1/2 * log((1 - errorWeighted) / errorWeighted);

		% Recompute data weights
		for j = 1:n
			dataWeights(j) = dataWeights(j) * exp(-classifierWeight * Y(j) * labelsPred(j));
		end

		% Renormalize data weights
		totalWeights = sum(dataWeights);
		dataWeights = dataWeights / totalWeights;

		% Remember the current classifier and its weight
		classifiers = [classifiers; classifier];
		classifiersWeights = [classifiersWeights; classifierWeight];
	end
end
