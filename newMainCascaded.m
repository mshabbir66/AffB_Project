clc
close all
clear all

% AffectDataSync = createAffectDataSync;
% save('./Dataset/AffectDataSync+sesNumber', 'AffectDataSync');

load ./Dataset/AffectDataSync+sesNumber

nfoldCV = 3;
nfold = 10;

% detection 1, recognition 2
classifierType=2;

% audio 1, video 2, feature fusion 3
modality=1;


if(classifierType==1)
    LAUGHTER = 1;
    BREATHING = 1;
    %OTHER = 1;
    REJECT = 2;
    axlabels={'Affect Burst','Reject'};
    saveName1='Detection';
else
    LAUGHTER = 1;
    BREATHING = 2;
    %OTHER = 1;
    REJECT = 3;
    axlabels={'Laughter','Breathing','Reject'};
    saveName1='Recognition';
end

%% label and feature extraction
LABEL=extractfield(AffectDataSync,'label')';
label = zeros(length(LABEL),1);
label(strcmp(LABEL,'Laughter')) = LAUGHTER;
label(strcmp(LABEL,'Breathing')) = BREATHING;
label(strcmp(LABEL,'REJECT')) = REJECT;
label(strcmp(LABEL,'Other')) = [];%OTHER

labelList = unique(label);
NClass = length(labelList);

if(modality==1) %audio
    for i=1:length(AffectDataSync)
        data(i,:)=extract_stats(AffectDataSync(i).data);
    end
    cRange=[-2 4 34];
    gRange=[-13 1 -7];
    saveName2='Audio';
elseif(modality==2) %video
    for i=1:length(AffectDataSync)
        data(i,:)=extract_stats(AffectDataSync(i).data);
    end
    cRange=[-2 4 34];
    gRange=[-10 1 -5];
    saveName2='Video';
else %fused
    for i=1:length(AffectDataSync)
        datatemp(i,:)=extract_stats(AffectDataSync(i).data);
        data(i,:)=[datatemp(i,:) extract_stats(AffectDataSync(i).data3d)];
    end
    cRange=[-2 4 46];
    gRange=[-14 1 -10];
    saveName2='Fused';
end

%% nfold test
CV(nfold).model=[];
len=length(label);
rand_ind = randperm(len);
figure;
for i=1:nfold % nfold test
  test_ind=rand_ind([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]');
  train_ind = [1:len]';
  train_ind(test_ind) = [];
  
  trainLabel=label(train_ind);
  trainData=data(train_ind,:);
  
  testLabel=label(test_ind);
  testData=data(test_ind,:);
  
  [CV(i).model, CV(i).bestParam, CV(i).grid ]= learn_on_trainingData(trainData, trainLabel, cRange, gRange, nfoldCV );
  
  [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, CV(i).model);
    acc(i).accuracy=accuracy(1);
    acc(i).testLabel = testLabel;
    acc(i).predict_label = predict_label;
    
    subplot(ceil(nfold/5),5,i);imagesc(CV(i).grid);
    
    disp(['done fold ', num2str(i)]);
end



acc = acc(~isnan(extractfield(acc,'accuracy')));

%% confusion matrix
predictLabels = extractfield(acc, 'predict_label');
testLabels = extractfield(acc, 'testLabel');
for i =1:NClass
    for j = 1:NClass
    ConfusionMatrix(i,j) = sum(predictLabels(testLabels==i)==j);
    end
end
ConfusionMatrixSensitivity = ConfusionMatrix./(sum(ConfusionMatrix,2)*ones(1,NClass));
ConfusionMatrixPrecision = ConfusionMatrix./(ones(NClass,1)*sum(ConfusionMatrix,1));

%% plot and metrics
figure;
bar3(ConfusionMatrix');
ax = gca;
set(ax,'XTickLabel',axlabels);
set(ax,'YTickLabel',axlabels);
xlabel('GT');
ylabel('P');

Precision = mean(diag(ConfusionMatrixPrecision));
Sensitivity = mean(diag(ConfusionMatrixSensitivity));

ave_acc=sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix));
title(['Confusion Matrix, ' ' Acc: ' num2str(100*ave_acc) '% Precision: ' num2str(100*mean(Precision)) '% Recall: ' num2str(100*mean(Sensitivity)) '%']);

saveName=['./EXPproper/' saveName1,saveName2];
saveas(gcf, saveName, 'fig');
save(saveName);