%% function forest = rfLearning(D, L, rndFeat)
%
%   grows a random forest classifier with the Forest-RI algorithm.
%	The resulting ensemble of classifiers is made up with L trees, each
%	of which being grown via the treeLearning function. This function
%	allows for the learning of a random tree if a rndFeat parameter is
%	given that defines the number of features to be randomly selected at
%	each node.
%
%	D : a PRTools dataset structure. It can be obtained from a (X,Y) couple
%		of matrices thanks to a call to 'prdataset(X,Y)'. In this case, X
%		is a matrix with each line corresponding to an input data point,
%		and Y is a vector with the corresponding outputs (true classes)
%	L : the number of tree to be grown in the forest
%	rndFeat : the number of random features at each node
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
function forest = rfLearning(D, L, rndFeat)

	islabtype(D, 'crisp');
	isvaldset(D, 1, 2); % at least 1 object per class, 2 classes
	% if rndFeat has not been given, it is set to d, the total number of available features
	if(~exist('rndFeat', 'var'))
		rndFeat = size(D.data, 2);
	end

	forest.learningMethod = 'Forest-RI';
	forest.nbClasses = length(unique(D.nlab));
	forest.nbTrees = L;
	forest.rndFeat = rndFeat;
	forest.trees = {};
	forest.boot = {};
	forest.oob = {};
	% Grow L trees and put them in a cell
	% Remember also data obtained from bootstrap and associated OOB data
	for i = 1:L
		% Extract data from the dataset
		X = D.data;
		Y = D.nlab;
		
		% Perform bootstrap
		[bag, oob] = drawBootstrap(size(X, 1), size(X, 1));
		
		% Remember bootstrap and OOB data
		D = prdataset(X(bag, :), Y(bag));
		ooD = prdataset(X(oob, :), Y(oob));
		forest.boot = [forest.boot; {D}];
		forest.oob = [forest.oob; {ooD}];
		
		% Perform the treelearning on the current dataset
		forest.trees = [forest.trees; {treeLearning(D, rndFeat)}];
	end
end
