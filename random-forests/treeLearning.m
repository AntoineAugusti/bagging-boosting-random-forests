%% function tree = treeLearning(D,rndFeat)
%
%   grow a (random) tree classifier with the CART method combined with
%   a Random Feature Selection procedure (no pruning available)
%	if 'rndFeat' is not given when the function is called, a regular decision
%	tree is grown. Otherwise, it specifies the number of random features
% 	to be pre-selected at each node, for finding the best splitting rule.
%
%
%	D : a PRTools dataset structure. It can be obtained from a (X,Y) couple
%		of matrices thanks to a call to 'prdataset(X,Y)'. in this case, X
%		is a matrix with each line corresponding to an input data point,
%		and Y is a vector with the corresponding outputs (true classes)
%
%	tree : the root node of the decision tree. This node structure may
%		represents an splitting node or a leaf node.
%		if a leaf node, contains 2 fields:
%			.split: a 'fake' splitting rule with a negative feature index
%			.proba: the a posteriori probabilities
%		if an internal node, contains 3 fields:
%			.split: the splitting rule returned by the randFeatSelection func
%			.left: the left node (recursive call)
%			.right: the right node (recursive call)
function tree = treeLearning(D,rndFeat)
	islabtype(D, 'crisp');
	isvaldset(D, 1, 2); % at least 1 object per class, 2 classes
	X = D.data;
	Y = D.nlab;
	C = length(unique(Y));
	% if rndFeat has not been given, it is set to d, the total number of available features
	if(~exist('rndFeat', 'var'))
		rndFeat = size(X, 2);
	end
	% grow the tree
	tree = makeTree(X, Y, C, rndFeat);
	tree.nbClasses = C;
	tree.rndFeat = rndFeat;
end

%%%%%
%	makeTree is a recursive function for growing a branch of a decision tree
%
%	X : matrix of the inputs at the current node
%	Y : column vector of the corresponding outputs (true classes)
%	C : number of classes
%	node : structure that represents either a leaf node or an internal node
%	if a leaf node, contains 2 fields:
%		.split: a 'fake' splitting rule with a negative feature index
%		.proba: the a posteriori probabilities
%	if an internal node, contains 3 fields:
%		.split: the splitting rule returned bu the findSplit function
%		.left: the left node (recursive call)
%		.right: the right node (recursive call)
function node = makeTree(X, Y, C, rndFeat)

	% 1 - find a splitting rule for the current node
	node.split = treeSplitting(X, Y, rndFeat);
	% 2 - if no splitting rule has been found, the returned node is a leaf
	if(node.split.feat < 0)
		node.proba = makeLeaf(X, Y, C);
	else
		% 3.a - split the node with the splitting rule found
		[XLeft, YLeft, XRight, YRight] = splitNode(X, Y, node.split);
		% 3.b - grow left and right branches recursively
		node.left = makeTree(XLeft,YLeft,C,rndFeat);
		node.right = makeTree(XRight,YRight,C,rndFeat);
	end
end

% Estimate a posteriori probabilities with proportion of instances belonging to each class.
function proba = makeLeaf(X, Y, C)
	proba = ones(1,C);
	for i=1:C
		proba(i) = sum(Y == i) / length(Y);
	end
end

% Split the current data set into two subsets according to the given splitting rule.
function [xl, yl, xr, yr] = splitNode(X, Y, split)
	leftMask = (X(:,split.feat) <= split.value);
	rightMask = ~leftMask;
	xl = X(leftMask,:);
	yl = Y(leftMask);
	xr = X(rightMask,:);
	yr = Y(rightMask);
end
