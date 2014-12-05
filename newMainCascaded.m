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
modality=2;


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

if(modality==1) %audio
    for i=1:length(AffectDataSync)
        data(i,:)=extract_stats(AffectDataSync(i).data);
    end
    cRange=[-2 4 34];
    gRange=[-13 1 -7];
    saveName2='Audio';
elseif(modality==2) %video
    for i=1:length(AffectDataSync)
        data(i,:)=extract_stats(AffectDataSync(i).data3d);
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

IDs=unique(extractfield(AffectDataSync,'id'));
len=length(IDs);
rand_ind = randperm(len);
load ('rand_ind.mat','rand_ind');
rand_id = IDs(rand_ind);

figure;
for i=1:nfold % nfold test
%  test_ind=rand_ind([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]');
%  train_ind = (1:len)';
%  train_ind(test_ind) = [];
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
  
  trainDataDetect=data(train_ind,:);
  
  testLabelDetect=label(test_ind);
  testLabelDetect(testLabelDetect == LAUGHTER) = 1;
  testLabelDetect(testLabelDetect == BREATHING) = 1;
  testLabelDetect(testLabelDetect == REJECT) = 2;
 
  testDataDetect=data(test_ind,:);
  
  trainLabelRec = label(train_ind);
  trainDataRec=data(train_ind,:);
  trainDataRec(trainLabelRec == REJECT,:) = [];
  trainLabelRec(trainLabelRec == REJECT) = [];

%   testLabelRec = label(test_ind);
%   testDataRec=data(test_ind,:);
%   testDataRec(testLabelRec == REJECT,:) = [];
%   testLabelRec(testLabelRec == REJECT) = [];
  
  [CV1(i).model, CV1(i).bestParam, CV1(i).grid ]= learn_on_trainingData(trainDataDetect, trainLabelDetect, cRange, gRange, nfoldCV );
  [CV2(i).model, CV2(i).bestParam, CV2(i).grid ]= learn_on_trainingData(trainDataRec, trainLabelRec, cRange, gRange, nfoldCV );

  [predict_label, accuracy, prob_values] = svmpredict(testLabelDetect, testDataDetect, CV1(i).model);
    acc1(i).accuracy=accuracy(1);
    acc1(i).testLabel = testLabelDetect;
    acc1(i).predict_label = predict_label;

   testLabelRec = label(test_ind);
   testLabelRec = testLabelRec(acc1(i).predict_label==1);
   testDataRec = data(test_ind,:);
   testDataRec = testDataRec(acc1(i).predict_label==1,:);
   
   
  [predict_label, accuracy, prob_values] = svmpredict(testLabelRec, testDataRec, CV2(i).model);
    acc2(i).accuracy=accuracy(1);
    acc2(i).testLabel = testLabelRec;
    acc2(i).predict_label = predict_label;
    
    acc3(i).testLabel = label(test_ind);

    
    subplot(ceil(nfold/5),5,i);imagesc(CV1(i).grid);
    drawnow;
    disp(['done fold ', num2str(i)]);
end

%acc = acc1(~isnan(extractfield(acc1,'accuracy')));

%% confusion matrix
for i = 1:length(acc1)
acc3(i).predict_label = acc1(i).predict_label;
acc3(i).predict_label(acc3(i).predict_label==2) = 3;
acc3(i).predict_label(acc3(i).predict_label==1) = acc2(i).predict_label;

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

saveName=['./EXPproper/Cascade' saveName1,saveName2];
saveas(gcf, saveName, 'fig');
save(saveName);