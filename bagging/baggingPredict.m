% function finalPredictions = baggingPredict(classifiers, Xtest, Ytest)
% 	Compute predictions for the bagging method
% 	Input:
% 	- classifiers: a cell array with multiple classifiers
% 	- Xtest: the test dataset
% 	- Ytest: labels of the test set
% 	Ouput:
% 	- finalPredictions: predictions made by the set of classifiers using bagging
function finalPredictions = baggingPredict(classifiers, Xtest, Ytest)
	dataTest = prdataset(Xtest, Ytest);
	preds = [];
	nbClassifiers = length(classifiers);

	% Get the predictions vector for each classifier
	for i = 1:nbClassifiers
		if (mod(i, 10) == 0)
			fprintf('Working on classifier %i / %i...\n', i, nbClassifiers);
		end
		classifier = classifiers{i};

		% Compute a classification dataset
		D = dataTest * classifier;
		% Get predicted labels
		labelsPred = D * labeld;
		% Remember predicted labels for bag i
		preds(i, :) = labelsPred;
	end

	% Find the most common prediction for each element
	finalPredictions = [];
	for j=1:size(preds, 2)
		finalPredictions = [finalPredictions; mode(preds(:, j))];
	end
end
