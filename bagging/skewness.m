% out = skewness(data)
% Compute the skewness over a dataset
function out = skewness(data)
	out = mean((data - mean(data) / std(data)).^3);
end
