% function EY = decisionStumpVal(classifier, X);
%   Input:
%	- classifier: struct from training
%	- X: N*M data matrix where N is the number of examples and M the number of features
%   Output:
%	- EY: N*1 estimated labels (-1 or 1)
function EY = decisionStumpVal(classifier, X);
	nX = size(X,1);
	nC = size(X,2);

	if classifier.sens == '<'
	    idPos = find(X(:, classifier.icarac) > classifier.seuil);
	else
	    idPos = find(X(:, classifier.icarac) < classifier.seuil);
	end

	EY = -ones(nX, 1);
	EY(idPos) = 1;
end
