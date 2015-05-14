% out = skewnessBootstrap(data, nbBags, nbFolds)
% 	Compute the mean skewness over a dataset using the Boostrap method
% 	Input:
% 	- data: the dataset
% 	- nbBags: the number of bags to create
% 	- nbFolds: the number of elements per bag
% 	Ouput:
% 	- out: the mean skewness for the dataset
function out = skewnessBootstrap(data, nbBags, nbFolds)
	skews = [];

	for i=1:nbBags
		[bag, oob] = drawBootstrap(length(data), nbFolds);
		skews = [skews;skewness(data(bag))];
	end

	out = mean(skews);
end
