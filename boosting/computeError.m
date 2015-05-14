% function err = computeError(preds, Y)
% 	Compute the number of errors made by a classifier as a percentage
% 	Input:
% 	- preds: predictions made by the classifier
% 	- Y: real labels
% 	Ouput:
% 	- err: the error made by the classifier as a percentage
function err = computeError(preds, Y)
	err = sum(preds ~= Y) / length(Y) * 100;
end
