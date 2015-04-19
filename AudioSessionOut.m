function [ConfusionMatrix]=AudioSessionOut(AffectDataSync)

nfoldCV=3;
%enum{
LAUGHTER = 1;
REJECT = 2;
%}

%% CV
 
LABEL=extractfield(AffectDataSync,'label')';
label = zeros(length(LABEL),1);
label(strcmp(LABEL,'Laughter')) = LAUGHTER;
label(strcmp(LABEL,'REJECT')) = REJECT;


for i=1:length(AffectDataSync)
    data(i,:)=extract_stats(AffectDataSync(i).data); %[datatemp(i,:) extract_stats(AffectDataSync(i).data3d)];
end


labelList = unique(label);
NClass = length(labelList);

cRange=[-2 4 46];
gRange=[-20 1 -1];

% %% Leave one Session out test
% 
for k=1:4
    
    testData=data(extractfield(AffectDataSync,'sesNumber')==k,:);
    testLabel=label(extractfield(AffectDataSync,'sesNumber')==k);
    
    trainData=data(extractfield(AffectDataSync,'sesNumber')~=k,:);
    trainLabel=label(extractfield(AffectDataSync,'sesNumber')~=k);
    
    
    [ model, bestParam, cv ] = learn_on_trainingData(trainData, trainLabel, cRange, gRange, nfoldCV, 0 );
    
    
    [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, model);
    
    acc(k).accuracy=accuracy(1);
    acc(k).testLabel = testLabel;
    acc(k).predict_label = predict_label;
    
    disp(['done with ', num2str(k)]);
end


acc = acc(~isnan(extractfield(acc,'accuracy')));
%ave = mean(extractfield(acc,'accuracy'));
%fprintf('Ave. Accuracy = %g%%\n', ave);


predictLabels = extractfield(acc, 'predict_label');
testLabels = extractfield(acc, 'testLabel');
for i =1:NClass
    for j = 1:NClass
    ConfusionMatrix(i,j) = sum(predictLabels(testLabels==i)==j);
    end
end

%% Plots and metrics!
TP=ConfusionMatrix(1,1);
FP=ConfusionMatrix(2,1);
TN=ConfusionMatrix(2,2);
FN=ConfusionMatrix(1,2);

ave_acc=100*(TP+TN)/(TP+FP+TN+FN);
precision=100*TP/(TP+FP);
recall=100*TP/(TP+FN);
%ave=100*sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix));
fprintf('Ave. Accuracy = %g%%\n', ave_acc);

figure;
set(gcf,'Position',[50 50 1200 600]);

%subplot(1,3,1)
bar3(ConfusionMatrix');
title(['Confusion Matrix, ' ' Acc: ' num2str(ave_acc) '% Precision: ' num2str(precision) '% Recall: ' num2str(recall) '%']);
ax=gca;
set(ax,'XTickLabel',{'Affect Burst','Reject'});
set(ax,'YTickLabel',{'Affect Burst','Reject'});
xlabel('GT');
ylabel('P')

end

