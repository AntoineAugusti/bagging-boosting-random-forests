%%  Finds and returns a splitting rule that allows to partition
%   an internal node of a decision tree, using the random feature selection
%	procedure. This function is meant to be used in a random tree growing procedure
%	in a Random Forest method.
%
%	X : matrix of the inputs (1 data point per line)
%	Y : column vector of the outputs (true classes)
%	split : a structure with 3 fields:
%		.crit : the gini value
%		.feat : index of the feature to be used in the splitting rule
%		.value : the corresponding splitting value
function split = treeSplitting(X, Y, rndFeat)

	% if all the data points belong to the same class, no need to look
	% for a splitting rule.
	c = max(Y);
	if(length(unique(Y)) == 1)
		split.feat = -1;
		return;
	end

	% Random feature selection
	[n, p] = size(X);
	permutations = randperm(p);
	featuresToUse = permutations(1:rndFeat);
	X = X(:, featuresToUse);

	% find the splitting rule that maximize the purity criterion
	[split.crit, split.feat, split.value] = splitCriterion(X, Y);
	% If a splitting rule has been found, map back
	% to the original feature to split
	if (split.feat > 0)
		split.feat = featuresToUse(split.feat);
	end
end

% Finds the best splitting rule according to the gini criterion
%
% 'a' is the matrix with the data subset at the current node.
% each column of this matrix contains the values of a candidate feature
% all the features of this matrix are tested, and the best splitting rules
% is returned with the corresponding gini value and splitting value
function [bestCrit, bestFeat, bestSplit] = splitCriterion(a, Y)
	[n,d] = size(a);
	c = max(Y);
	crits = []; splits = [];
	dist0 = zeros(1, c);
	for i=1:c
		dist0(i) = sum(Y==i);
	end
	g0 = gini(dist0);

	for k=1:d
		[sX,sIdx] = sort(a(:,k));
		sY = Y(sIdx);
		distl = zeros(1,c);
		distr = dist0;
		nl = 0; nr = n;
		ming = inf; idx = 1;
		for i = 1:n-1
			distl(sY(i)) = distl(sY(i))+1;
			distr(sY(i)) = distr(sY(i))-1;
			nl = nl + 1;
			nr = nr - 1;
			if(sX(i) ~= sX(i+1))
				gl = gini(distl);
				gr = gini(distr);
				g = (nl*gl + nr*gr);
				if(g <= ming)
					ming = g;
					idx = i;
				end
			end
		end
		crits = [crits ming/n];
		splits = [splits mean(sX(idx:idx+1))];
	end

	[bestCrit,bestFeat] = min(crits);
	bestCrit = g0 - bestCrit;
	bestSplit = splits(bestFeat);
	if (bestCrit <= 0.0)
		bestFeat = -1;
	end
end

function G = gini(distrib)
	fy = distrib / sum(distrib);
	G = 1.0 - sum(fy.^2);
end
