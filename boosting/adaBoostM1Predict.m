% preds = adaBoostM1Predict(X, nbClasses, classifiers, classifiersWeights)
%	Get predictions for a test set from a set of classifiers using the AdaBoost M1
%	algorithm
%	Input:
%	- X: N*M data matrix where N is the number of examples and M the number of features
%	- nbClasses: the number of labels
% 	- classifiers: a cell array of classifiers. The number of classifiers
% 	corresponds to the number of iterations
% 	- classifiersWeights: a vector of weights for each classifier
%	Ouput:
%	- preds: N*1 matrix of labels (ranging from 1 to nbClasses) for each element of the test set
function preds = adaBoostM1Predict(X, nbClasses, classifiers, classifiersWeights)

	classifiersPredictions = [];
	dataset = prdataset(X);
	for i = 1:length(classifiers)
		classifier = classifiers{i};
		% Compute a classification dataset
		D = dataset * classifier;
		% Get predicted labels
		classifierPredictions = D * labeld;
		% Remember predictions for the current classifier
		classifiersPredictions = [classifiersPredictions classifierPredictions];
	end

	preds = zeros(size(X, 1), 1);
	for i = 1:size(X, 1)
		bestThetaForClasses = -inf;
		bestClass = 0;
		mySum = 0;
		data = classifiersPredictions(i, :);

		% Try to find the best class
		for currentClass = 1:nbClasses
			mySum = (data == currentClass) * classifiersWeights;
			if (mySum > bestThetaForClasses)
				bestThetaForClasses = mySum;
				bestClass = currentClass;
			end
		end

		preds(i) = bestClass;
	end
end
