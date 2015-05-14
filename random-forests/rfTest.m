% function res = treeTest(D, forest)
% 	Test a random forest classifier with a test set
% 	Input:
% 	- D: the PRTools dataset
% 	- forest: the random forest coming from the rfLearning function
% 	Ouput:
% 	res: a structure that contains the following fields:
% 	- proba: a row vector for each element with a "1" for the predicted class. Example: [0 0 1 0] for class 3
% 	- pred: the predicted class for each element
% 	- nbErrors: the number of errors made
% 	- confm: the confusion matrix
% 	- nbTestInstances: the number of elements tested
% 	- errRate: the error rate on the test set, between 0 and 1
function res = rfTest(D, forest)
	res.proba = rfPredict(D.data, forest);
	[tmp, res.pred] = max(res.proba, [], 2);
	res.confm = zeros(forest.nbClasses, forest.nbClasses);
	for i = 1:length(res.pred)
		res.confm(D.nlab(i), res.pred(i)) = res.confm(D.nlab(i), res.pred(i)) + 1;
	end
	res.nbErrors = sum(res.pred~=D.nlab);
	res.nbTestInstances = size(D.data, 1);
	res.errRate = res.nbErrors / size(D.data, 1);
end
