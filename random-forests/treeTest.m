% function res = treeTest(D, tree)
% 	Test a tree classifier with a test set
% 	Input:
% 	- D: the PRTools dataset
% 	- tree: the tree coming from the treeLearning function
% 	Ouput:
% 	res: a structure that contains the following fields:
% 	- proba: a row vector for each element with a "1" for the predicted class. Example: [0 0 1 0] for class 3
% 	- pred: the predicted class for each element
% 	- nbErrors: the number of errors made
% 	- nbTestInstances: the number of elements tested
% 	- errRate: the error rate on the test set, between 0 and 1
function res = treeTest(D, tree)
	res.proba = treePredict(D.data, tree);
	[tmp, res.pred] = max(res.proba, [], 2);
	res.nbErrors = sum(res.pred ~= D.nlab);
	res.nbTestInstances = size(D.data, 1);
	res.errRate = res.nbErrors / size(D.data, 1);
end
