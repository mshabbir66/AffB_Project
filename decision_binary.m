%% forest binary
clc
close all
clear all

%enum{
LAUGHTER = 1;
BREATHING = 2;
OTHER = 3;
REJECT = 4;
%}

%load AffBinaryStats750ms66ol_D
addpath('./sherwood-classify-matlab')
%load stats_750_250_0_D

% labelsNA = ones(size(nonAffStats,1),1)+1;
% labelsA = ones(size(AffStats,1),1);
% trainData = [nonAffStats;AffStats];
% trainLabel = [labelsNA;labelsA];
% ind = randperm(length(trainData))';
% trainData = trainData(ind,:);
% trainLabel = trainLabel(ind);
%%
load ./Dataset/AffectData

%trainLabel=extractfield(AffectData,'label')';
LABEL=extractfield(AffectData,'label')';
trainLabel = zeros(length(LABEL),1);
trainLabel(strcmp(LABEL,'Laughter')) = LAUGHTER;
trainLabel(strcmp(LABEL,'Breathing')) = BREATHING;
trainLabel(strcmp(LABEL,'Other')) = OTHER;
trainLabel(strcmp(LABEL,'REJECT')) = REJECT;



%trainData=zeros(length(AffectData),length(AffectData(1).data));
for i=1:length(AffectData)
    trainData(i,:) = extract_stats(AffectData(i).data);
end


trainData = single(trainData);
trainLabel = uint8(trainLabel);


settings.MaxDecisionLevels = 10;

% Number of candidate feature response functions per split node
% Default: 10.
settings.NumberOfCandidateFeatures = 10;

% Optimal entropy split is determined by thresholding on 
% NumberOfCandidateThresholdsPerFeature equidistant points
% Default 10.
settings.NumberOfCandidateThresholdsPerFeature = 10;

% Number of trees in the forest
% Default: 30.
settings.NumberOfTrees = 30;

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
settings.forestName = 'temp.bin';

%sherwood_train(trainData', trainLabel, settings);
%probabilities = sherwood_classify(trainData', settings);
%predictLabels = ones(length(probabilities),1);

%predictLabels(probabilities(1,:) < 0.5) = 2;

%disp(num2str(sum((trainLabel==predictLabels))/length(trainLabel)));
%ac = get_cv_ac_bin_decisionBin(trainLabel, trainData, settings, 3);

 bestcv = 0;
 i =1; j =1;
 for MaxDecisionLevels = 11:15,
     for NumberOfTrees = 80:10:150
         settings.MaxDecisionLevels = MaxDecisionLevels;
         settings.NumberOfTrees = NumberOfTrees;
         cv(i,j) = get_cv_ac_bin_decisionBin(trainLabel, trainData, settings, 3);
          if (cv(i,j) >= bestcv),
             bestcv = cv(i,j); bestMaxDecisionLevels = MaxDecisionLevels; bestNumberOfTrees = NumberOfTrees;
         end
         fprintf('%g %g %g (best MDL=%g, NOT=%g, rate=%g)\n', MaxDecisionLevels, NumberOfTrees, cv(i,j), bestMaxDecisionLevels, bestNumberOfTrees, bestcv);
         j = j+1;
     end
     j =1;
     i = i + 1;
 end
 
 save decision_best_Params_750_250_0_D cv bestcv bestMaxDecisionLevels bestNumberOfTrees;
 
