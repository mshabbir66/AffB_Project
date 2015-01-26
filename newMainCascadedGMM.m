clc
close all
clear all

% AffectDataSync = createAffectDataSync;
% save('./Dataset/AffectDataSync+sesNumber', 'AffectDataSync');

load ./Dataset/AffectDataSyncNew

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
AffectDataSync(strcmp(LABEL,'Other')) = [];

labelList = unique(label);
NClass = length(labelList);
com=8;

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
foldsDet(nfold).model=[];
foldsRec(nfold).model=[];
acc3(nfold).testLabel=[];

IDs=unique(extractfield(AffectDataSync,'id'));
len=length(IDs);
rand_ind = randperm(len);
load ('rand_ind.mat','rand_ind');
rand_id = IDs(rand_ind);

for i=1:nfold % nfold test
  train_ind=[];test_ind=[];
  test_id=rand_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]');
  train_id = rand_id;
  train_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]) = [];
  
  for k=1:length(train_id)
      train_ind=[train_ind;find(extractfield(AffectDataSync,'id')==train_id(k))'];
  end
  for k=1:length(test_id)
      test_ind=[test_ind;find(extractfield(AffectDataSync,'id')==test_id(k))'];
  end


  trainLabelDetect=label(train_ind);
  trainLabelDetect(trainLabelDetect == LAUGHTER) = 1;
  trainLabelDetect(trainLabelDetect == BREATHING) = 1;
  trainLabelDetect(trainLabelDetect == REJECT) = 2;
  
  trainDataDetect=AffectDataSync(train_ind);
  
  testLabelDetect=label(test_ind);
  testLabelDetect(testLabelDetect == LAUGHTER) = 1;
  testLabelDetect(testLabelDetect == BREATHING) = 1;
  testLabelDetect(testLabelDetect == REJECT) = 2;
 
  testDataDetect=AffectDataSync(test_ind);
  
  trainLabelRec = label(train_ind);
  trainDataRec=AffectDataSync(train_ind);
  trainDataRec(trainLabelRec == REJECT) = [];
  trainLabelRec(trainLabelRec == REJECT) = [];
  
  
  [ foldsDet(i).model ] = trainGMM(trainDataDetect, trainLabelDetect, com);
  [ foldsRec(i).model ] = trainGMM(trainDataRec, trainLabelRec, com);

  
  Pos=zeros(length(testDataDetect),2);
    for j=1:length(testDataDetect) 
        for class=1:2
            [~,Pos(j,class)] = posterior(foldsDet(i).model(class).obj,testDataDetect(j).data);           
        end
    end    
    [v ix] = sort(Pos,2);
    %acc=sum(ix(:,1)==testLabel)/length(testLabel);
    foldsDet(i).testLabel = testLabelDetect;
    foldsDet(i).predict_label = ix(:,1);
    foldsDet(i).prob_values = Pos;

   testLabelRec = label(test_ind);
   testLabelRec = testLabelRec(foldsDet(i).predict_label==1);
   testDataRec = AffectDataSync(test_ind);
   testDataRec = testDataRec(foldsDet(i).predict_label==1);
   
   
   Pos=zeros(length(testDataRec),NClass);
    for j=1:length(testDataRec) 
        for class=1:NClass
            [~,Pos(j,class)] = posterior(foldsRec(i).model(class).obj,testDataRec(j).data);           
        end
    end    
    [v ix] = sort(Pos,2);
    %acc=sum(ix(:,1)==testLabel)/length(testLabel);
    foldsRec(i).testLabel = testLabelRec;
    foldsRec(i).predict_label = ix(:,1);
    foldsRec(i).prob_values = Pos;
    
    acc3(i).testLabel = label(test_ind);

    
    disp(['done fold ', num2str(i)]);
end

%acc = acc1(~isnan(extractfield(acc1,'accuracy')));

%% confusion matrix
for i = 1:length(foldsDet)
acc3(i).predict_label = foldsDet(i).predict_label;
acc3(i).predict_label(acc3(i).predict_label==2) = 3;
acc3(i).predict_label(acc3(i).predict_label==1) = foldsRec(i).predict_label;

end

predictLabels = extractfield(acc3, 'predict_label');
testLabels = extractfield(acc3, 'testLabel');

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

saveName=['./EXPproper/GMMCascade' saveName1,saveName2];
saveas(gcf, saveName, 'fig');
save(saveName);



