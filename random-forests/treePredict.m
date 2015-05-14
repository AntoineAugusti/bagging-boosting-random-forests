% function proba = treePredict(X, tree)
% 	Compute the predicted class for each element of the test set
% 	Input:
% 	- X: the PRTools dataset
% 	- tree: the tree, coming from the treeLearning function
% 	Output:
% 	- proba: a row vector for each element with a "1" for the predicted class. Example: [0 0 1 0] for class 3
function proba = treePredict(X, tree)
	for i=1:size(X, 1)
		proba(i, :) = recursPredict(X(i, :), tree);
	end
end

function prob = recursPredict(x, tree)
	if (tree.split.feat < 0)
		prob = tree.proba;
	else
		if(x(tree.split.feat) <= tree.split.value)
			prob = recursPredict(x, tree.left);
		else
			prob = recursPredict(x, tree.right);
		end
	end
end
