function [ model, bestParam, cv ] = learn_on_trainingData(trainData, trainLabel, cRange, gRange, nfoldCV, probEnable )
%UNTÝTLED2 Summary of this function goes here
%   Detailed explanation goes here
% % #######################
% % Parameter selection using 3-fold cross validation
% % #######################
bestcv = 0;
i =1; j =1;
for log2c = cRange(1):cRange(2):cRange(3),
    for log2g = gRange(1):gRange(2):gRange(3),
        cmd = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv(i,j) = get_cv_ac_bin(trainLabel, trainData, cmd, nfoldCV);
        if (cv(i,j) >= bestcv),
            bestcv = cv(i,j); bestc = 2^log2c; bestg = 2^log2g;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv(i,j), bestc, bestg, bestcv);
        j = j+1;
    end
    j =1;
    i = i + 1;
end

%% #######################
% % Train the SVM in one-vs-rest (OVR) mode
% % #######################
 if(probEnable)
    bestParam = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg) ' -b 1'];
 else
    bestParam = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg)];
 end
    
    model = svmtrain(trainLabel, trainData, bestParam);

end

