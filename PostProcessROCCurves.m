%% post process
clear all
load('NewResultsFeatureFusedWithProbabilities.mat')


A = struct2cell(acc);
probs = cell2mat(A(4,:)');
scores = zeros(length(probs),2);
scores(:,1)  = probs(:,1) + probs(:,2);
scores(:,2)   = probs(:,3);
labels = extractfield(acc,'testLabel');
labels(labels==2) = 1;
labels(labels==3) = 2;



[X,Y,T,AUC] = perfcurve(labels,scores(:,1),1);
roc.X=X;
roc.Y=Y;
roc.T=T;
roc.AUC=AUC;
figure(1);
plot(X,Y)
xlabel('False positive rate'); ylabel('True positive rate')
title(['ROC for classification, AUC: ' num2str(AUC)])

[X,Y] = perfcurve(labels,scores(:,1),1,'xCrit','prec','yCrit','reca');
figure(2);
plot(X,Y);
xlabel('Precision'); ylabel('Recall');
title('ROC for classification');