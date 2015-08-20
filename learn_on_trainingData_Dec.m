function [settings, cv ] = learn_on_trainingData_Dec(trainData, trainLabel)

settings.MaxDecisionLevels = 20;
settings.NumberOfCandidateFeatures = 30;
settings.NumberOfCandidateThresholdsPerFeature = 10;
settings.NumberOfTrees = 30;
settings.verbose = false;
settings.WeakLearner = 'random-hyperplane';
settings.MaxThreads =  feature('NumThreads');
settings.forestName = 'temp.bin';

bestcv = 0;
i =1; j =1;
for MaxDecisionLevels = 5:2:15,
    for NumberOfTrees = 40:20:200
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

sherwood_train(trainData', trainLabel, settings);
