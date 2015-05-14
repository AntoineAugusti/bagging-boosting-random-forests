% function err = baggingError(baggingPredictions, Ytest)
% 	Count the number of errors made by the bagging method
% 	Input:
% 	- baggingPredictions: predictions made by the bagging method
% 	- Ytest: real labels from the test set
%	Output:
%	- err: the error made, between 0 and 1
function err = baggingError(baggingPredictions, Ytest)
	% Count the number of errors made
	err = sum(baggingPredictions ~= Ytest) / length(baggingPredictions);
end
