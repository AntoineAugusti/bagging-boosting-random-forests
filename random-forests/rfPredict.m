%% function proba = rfPredict(X, forest)
%   Grows a random forest classifier with the Forest-RI algorithm.
%	The resulting ensemble of classifiers is made up with L trees, each
%	of which being grown via the treeLearning function. This function
%	allows for the learning of a random tree if a rndFeat parameter is
%	given that defines the number of features to be randomly selected at
%	each node.
%
%	X : a new value
%
%	forest : a structure that contains 6 fields:
%		.learningMethod = "Forest-RI"
%		.nbTrees = L
%		.rndFeat = rndFeat
%		.nbClasses = The number of classes
%		.trees : a row cellarray with L cells, each of which containing
%				 a decision	tree classifier (see help treeLearning)
%		.boot : a row cellarray with L cells, each of which containing
%				a bootstrap sample
%		.oob : a row cellarray with L cells, each of which containing
%				the corresponding out-of-bag sets
%	proba : a probality vector for each class [0.8; 0.5; 0.2] => we will predict the
% 			class 1
function proba = rfPredict(X, forest)
	probas = [];

	for line = 1:size(X, 1)
		tmp = [];
		for i = 1:forest.nbTrees
			proba = treePredict(X(line, :), forest.trees{i});
			tmp = [tmp; proba];
		end
		probas(line, :) = mean(tmp);
	end

	proba = probas;
end
