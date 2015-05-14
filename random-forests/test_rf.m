clear all;
close all;
clc;

data = load('datasets/diabetes.mat');
D = prdataset(data.X, data.Y);
[Dr, Ds] = gendat(D, 0.66);

merrK = [];
stderrK = [];
% Try several value of random features
for k = 1:2:7
	fprintf('-- Trying with %d random features\n', k);
	err = [];
	for i=1:5
		fprintf('   Run %d out of 5\n', i);
		tic; forest = rfLearning(Dr, 50, k);
		res = rfTest(Ds, forest); toc;
		err = [err res.errRate];
	end
	merrK = [merrK mean(err)]
	stderrK = [stderrK std(err)]
end

X = [1:2:7];
errorbar(X, merrK, stderrK);

% RF from the PRTools
% very slow!!!
%tic; w1 = randomforestc(Dr,50,1); toc;
%err = testc(Ds,w1)
