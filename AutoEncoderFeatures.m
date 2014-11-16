clc
close all
clear all

load ForAutoEncoder;



extractedData = extractfield(Stats,'data');
featureLength = length(extractedData)/length(Stats);
dataSamples = reshape(extractedData,featureLength,length(Stats));
meanDataSamples = repmat(mean(dataSamples,2),1,length(Stats));
dataSamples = dataSamples - meanDataSamples;
scaleFactor = max(abs(dataSamples),[],2);
dataSamples = dataSamples./repmat(scaleFactor,1,length(Stats));

visibleSize = size(dataSamples,1);   % number of input units 
hiddenSize = floor(size(dataSamples,1)/16);     % number of hidden units 
sparsityParam = 0.01;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term     
theta = initializeParameters(hiddenSize, visibleSize);

addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, dataSamples), ...
                              theta, options);

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);


