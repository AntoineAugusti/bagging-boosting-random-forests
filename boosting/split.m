% [xapp, yapp, xtest, ytest] = split(x, y, ratio)
%	Split a dataset and its labels into a training and test set
%	Input:
%	- x: the dataset
%	- y: the associated labels
%	- ratio: a real between 0 and 1. 0.4 means 40% for training, 60% for test
%	Output:
%	- xapp: the training dataset
%	- yapp: labels for the training dataset
%	- xtest: the test dataset
%	- ytest: labels for the test dataset
function [xapp, yapp, xtest, ytest] = split(x, y, ratio)

	classcode = unique(y);

	xapp = [];
	yapp = [];
	xtest = [];
	ytest = [];

	for numclass = 1:length(classcode)
		indclass = find(y==classcode(numclass));
		Ni  = length(indclass);
		aux = randperm(Ni);
		auxapp = aux(1: ceil(ratio*Ni));
		auxtest = aux(ceil(ratio*Ni)+1:end);
		xapp = [xapp; x(indclass(auxapp),:)];
		yapp = [yapp; y(indclass(auxapp))];

		xtest = [xtest; x(indclass(auxtest),:)];
		ytest = [ytest; y(indclass(auxtest))];
	end
end
