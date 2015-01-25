clear all;
clc
close all


% AffectDataSync = createAffectDataSync;
% save('./Dataset/AffectDataSync+sesNumber', 'AffectDataSync');

load ./Dataset/AffectDataSyncNew

% Removing Other class
AffectDataSync(strcmp(extractfield(AffectDataSync,'label'),'Other'))=[];

nfoldCV=3;
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

labelList = unique(label);
NClass = length(labelList);
comRange=[1,1,3];

if(modality==1) %audio
    saveName2='Audio';
elseif(modality==2) %video
    for i=1:length(AffectDataSync)
        AffectDataSync(i).data=AffectDataSync(i).data3d;
    end
    saveName2='Video';
else %fused
    for i=1:length(AffectDataSync)
        AffectDataSync(i).data=[AffectDataSync(i).data AffectDataSync(i).data3d];
    end
    saveName2='Fused';
end


%% nfold test
folds(nfold).bestCom=[];
IDs=unique(extractfield(AffectDataSync,'id'));
len=length(IDs);
load rand_ind.mat%rand_ind = randperm(len);
rand_id = IDs(rand_ind);
parfor i=1:nfold % nfold test
  train_ind=[];test_ind=[];
  test_id=rand_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]');
  train_id = rand_id;
  train_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]) = [];
  
  for k=1:length(train_id)
      train_ind=[train_ind;find(extractfield(AffectDataSync,'id')==train_id(k))'];
  end
  trainData=AffectDataSync(train_ind,:);
  trainLabel=label(train_ind);
  
  for k=1:length(test_id)
      test_ind=[test_ind;find(extractfield(AffectDataSync,'id')==test_id(k))'];
  end
  testData=AffectDataSync(test_ind,:);
  testLabel=label(test_ind);
  
  [ model, bestCom, cv ] = CVtrainGMM(trainData, trainLabel, comRange, nfoldCV);
    folds(i).model=model;
    folds(i).bestCom=bestCom;
    folds(i).cv=cv;
Pos=zeros(length(testData),NClass);
for j=1:length(testData) 
    for class=1:NClass
        [~,Pos(j,class)] = posterior(model(class).obj,testData(j).data);           
    end
end    
    [v ix] = sort(Pos,2);
    %acc=sum(ix(:,1)==testLabel)/length(testLabel);
    folds(i).testLabel = testLabel;
    folds(i).predict_label = ix(:,1);
    folds(i).prob_values = Pos;
    disp(['done fold ', num2str(i)]);
end


%% confusion matrix
predictLabels = extractfield(folds, 'predict_label');
testLabels = extractfield(folds, 'testLabel');
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

saveName=['./EXPproper/GMM' saveName1,saveName2];
saveas(gcf, saveName, 'fig');
save(saveName);