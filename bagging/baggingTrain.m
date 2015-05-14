% function classifiers = baggingTrain(Xapp, Yapp, ratioTest, methodToClassify, nbBags, nbFolds)
%	Perform the bagging algorithm on a dataset
% 	Input:
% 	- Xapp: the whole dataset (train + test)
% 	- Yapp: the whole labels (train + test)
%	- methodToClassify: tell which classifier we will use. Supported values: 'knn' or 'tree'
%	- nbBags: the number of bags
%	- nbFolds: the number of elements per bag
%	Output:
%	- classifiers: a cell array with multiple classifiers
function classifiers = baggingTrain(Xapp, Yapp, methodToClassify, nbBags, nbFolds)
	classifiers = cell(nbBags, 1);

	for i = 1:nbBags
		if (mod(i, 10) == 0)
			fprintf('Working on bag %i / %i...\n', i, nbBags);
		end
		% Get the data for our current bag using Bootstrap
		[bag, oob] = drawBootstrap(length(Xapp), nbFolds);

		% Train the classifier for the current bag
		dataForBag = prdataset(Xapp(bag, :), Yapp(bag));
		if (strcmp(methodToClassify, 'tree'))
			classifier = treec(dataForBag);
		else
			classifier = knnc(dataForBag, 3);
		end
		% Store the classifier
		classifiers{i} = classifier;
	end
end
