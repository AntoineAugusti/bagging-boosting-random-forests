clear all;
close all;
clc;

synth4 = load('datasets/synth4.mat');
D = prdataset(synth4.X, synth4.Y);
[Dr, Ds] = gendat(D, 0.3);

fprintf('Growing a PRTools tree classifier\n');
tic; w1 = treec(Dr); toc;
fprintf('Growing a decision tree with the treeLearning function\n');
tic; tree = treeLearning(Dr); toc;

% Compare errors
res = treeTest(Ds, tree);
fprintf('Error with our tree %f %% \n', res.errRate * 100);
err = testc(Ds, w1);
fprintf('Error with the tree from PRTools %f %% \n', err * 100);
