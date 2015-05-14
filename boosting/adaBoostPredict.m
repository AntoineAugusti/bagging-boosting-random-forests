% function preds = adaBoostPredict(X, classifiers, classifiersWeights)
%	Get predictions for a test set thanks to a set of classifiers using
%	the AdaBoost algorithm
%	Input:
%	- X: N*M data matrix where N is the number of examples and M the number of features
% 	- classifiers: a cell array of classifiers. Can be obtained thanks to adaBoostTrain
% 	- classifiersWeights: a vector of weights for each classifier. Can be obtained thanks to adaBoostTrain
%	Ouput:
%	- preds: N*1 matrix of labels (1 or -1) for each element of the test set
function preds = adaBoostPredict(X, classifiers, classifiersWeights)

	classifiersPredictions = [];
	% Get predictions for each classifier
	for i = 1:length(classifiers)
		classifiersPrediction = decisionStumpVal(classifiers(i), X);
		classifiersPredictions = [classifiersPredictions classifiersPrediction];
	end
	% Multiply prediction for each classifier by its weight
	out = classifiersPredictions * classifiersWeights;
	preds = sign(out);
end
