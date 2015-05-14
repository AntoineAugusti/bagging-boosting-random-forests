% function [bag, oob] = drawBootstrap(nbData, nbFolds)
% 	Find indices to use during bootstrap
% 	Input:
% 	- nbData: the number of elements in the dataset
% 	- nbFolds: the number of elements to put in the bag
% 	Output:
% 	- bag: indices of elements to put in the bag
% 	- oob: indices of elements outside the bag
%
function [bag, oob] = drawBootstrap(nbData, nbFolds)
	bag = randi([1 nbData], nbFolds, 1);
	oob = setdiff([1:nbData], bag)';
end
