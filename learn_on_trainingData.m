function [ model, bestParam, cv ] = learn_on_trainingData(trainData, trainLabel, cRange, gRange, nfoldCV, probEnable,k )
%UNTÝTLED2 Summary of this function goes here
%   Detailed explanation goes here
% % #######################
% % Parameter selection using 3-fold cross validation
% % #######################

log2c = cRange(1):cRange(2):cRange(3);
log2g = gRange(1):gRange(2):gRange(3);
cv=zeros(length(log2c),length(log2g));

 if(probEnable)
     for i=1:length(log2c)
         for j=1:length(log2g)
             cmd = ['-q -c ', num2str(2^log2c(i)), ' -g ', num2str(2^log2g(j)), ' -b 1'];
             cv(i,j) = get_cv_ac_bin_probabilistic(trainLabel, trainData, cmd, nfoldCV);
             fprintf('%g %g %g Fold:%d\n', log2c(i), log2g(j), cv(i,j),k);
         end
     end 
 else
     for i=1:length(log2c)
         parfor j=1:length(log2g)
             cmd = ['-q -c ', num2str(2^log2c(i)), ' -g ', num2str(2^log2g(j))];
             cv(i,j) = get_cv_ac_bin(trainLabel, trainData, cmd, nfoldCV);
             fprintf('%g %g %g Fold:%d\n', log2c(i), log2g(j), cv(i,j),k);
         end
     end
 end

[y, ind]=max(cv);
[y1, indj]=max(y);
indi=ind(indj);
bestcv = cv(indi,indj); bestc = 2^log2c(indi); bestg = 2^log2g(indj);

         
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

