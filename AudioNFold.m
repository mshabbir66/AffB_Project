function [ConfusionMatrix,acc]=AudioNFold(AffectDataSync)

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


parfor i=1:length(AffectDataSync)
    data(i,:)=extract_stats(AffectDataSync(i).data); %[datatemp(i,:) extract_stats(AffectDataSync(i).data3d)];
    disp(['Extracted stats for sample ', num2str(i)]);
end


labelList = unique(label);
NClass = length(labelList);

cRange=[-2 4 42];
gRange=[-15 1 -9];

% %% Leave one Session out test
% 
sesNum=extractfield(AffectDataSync,'sesNumber');
gender=extractfield(AffectDataSync,'gender');
for i=1:length(sesNum)
temp(i)=sesNum{i};
end

cnt=0;
genCode=['M','F'];
for k=unique(temp)
    for l=1:2
        
        mask =(temp==k)&strcmp(gender,genCode(l));
        testData=data(mask,:);
        testLabel=label(mask);
        if(isempty(testLabel))
            continue;
        end
        cnt=cnt+1;
        trainData=data(~mask,:);
        trainLabel=label(~mask);


        [ model, bestParam, cv ] = learn_on_trainingData(trainData, trainLabel, cRange, gRange, nfoldCV, 0 );


        [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, model);

        acc(cnt).accuracy=accuracy(1);
        acc(cnt).testLabel = testLabel;
        acc(cnt).predict_label = predict_label;

        disp(['done with ', num2str(cnt)]);
    end
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

