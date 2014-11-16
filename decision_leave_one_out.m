
clc;
close all; 
clear all;
addpath('./sherwood-classify-matlab')
% Load Data
load stats_750_250_0_D;
load decision_best_Params_750_250_0_D
% Best Params

settings.MaxDecisionLevels = bestMaxDecisionLevels;

% Number of candidate feature response functions per split node
% Default: 10.
settings.NumberOfCandidateFeatures = 10;

% Optimal entropy split is determined by thresholding on 
% NumberOfCandidateThresholdsPerFeature equidistant points
% Default 10.
settings.NumberOfCandidateThresholdsPerFeature = 10;

% Number of trees in the forest
% Default: 30.
settings.NumberOfTrees = bestNumberOfTrees;

% Default: false.
settings.verbose = false;

% Determine which weak learner to be used as a split function.
% Options {random-hyperplane, axis-aligned-hyperplane}
% Default: axis-aligned-hyperplane
settings.WeakLearner = 'random-hyperplane'; 

% Thread(s) used when training and testing.
% Default: 1
settings.MaxThreads =  feature('NumThreads');

% The serialized forest will be saved and loaded from this filename
% Default: forest.bin
settings.forestName = 'temp_leaveOneOut.bin';

label=extractfield(Stats,'label')';
data=zeros(length(Stats),length(Stats(1).data));
for i=1:length(Stats)
    data(i,:)=extractfield(Stats(i),'data');
end

for i = 1:length(unique(extractfield(Stats,'id')))
    testData=single(data(extractfield(Stats,'id')==i,:));
    testLabel=uint8(label(extractfield(Stats,'id')==i));
    
    trainData=single(data(extractfield(Stats,'id')~=i,:));
    trainLabel=uint8(label(extractfield(Stats,'id')~=i));
    
    sherwood_train(trainData', trainLabel, settings);
  
    probabilities = sherwood_classify(testData', settings);
    predictLabels = ones(size(probabilities,2),1);
    predictLabels(probabilities(1,:) < 0.5) = 2;
    
    acc(i)=sum((testLabel==predictLabels))/length(testLabel);
    
    disp(['done with ', num2str(i)]);
end

