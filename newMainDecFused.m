clc
close all
clear all

% AffectDataSync = createAffectDataSync;
% save('./Dataset/AffectDataSync+sesNumber', 'AffectDataSync');

load ./Dataset/AffectDataSync+sesNumber

% Removing Other class
AffectDataSync(strcmp(extractfield(AffectDataSync,'label'),'Other'))=[];

nfoldCV = 3;
nfold = 10;



% detection 1, recognition 2
classifierType=1;


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
%label(strcmp(LABEL,'Other')) = OTHER

labelList = unique(label);
NClass = length(labelList);


for i=1:length(AffectDataSync)
    data(i,:)=extract_stats(AffectDataSync(i).data);
end
cRange=[-2 4 34];
gRange=[-13 1 -8];

for i=1:length(AffectDataSync)
    data3d(i,:)=extract_stats(AffectDataSync(i).data3d);
end
cRange3d=[-2 4 34];
gRange3d=[-11 1 -7];

saveName2='DecFused';


%% nfold test
CV(nfold).model=[];
CV3d(nfold).model=[];
IDs=unique(extractfield(AffectDataSync,'id'));
len=length(IDs);
load rand_ind.mat%rand_ind = randperm(len);
rand_id = IDs(rand_ind);
prob=[];prob3d=[];
for i=1:nfold % nfold test
  train_ind=[];test_ind=[];
  test_id=rand_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]');
  train_id = rand_id;
  train_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]) = [];
  
  for k=1:length(train_id)
      train_ind=[train_ind;find(extractfield(AffectDataSync,'id')==train_id(k))'];
  end
  trainData=data(train_ind,:);
  trainData3d=data3d(train_ind,:);
  trainLabel=label(train_ind);
  
  for k=1:length(test_id)
      test_ind=[test_ind;find(extractfield(AffectDataSync,'id')==test_id(k))'];
  end
  testData=data(test_ind,:);
  testData3d=data3d(test_ind,:);
  testLabel=label(test_ind);
  
  [CV(i).model, CV(i).bestParam, CV(i).grid ]= learn_on_trainingData(trainData, trainLabel, cRange, gRange, nfoldCV, 1);
  [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, CV(i).model,'-b 1');
    acc(i).accuracy=accuracy(1);
    acc(i).testLabel = testLabel;
    acc(i).predict_label = predict_label;
    acc(i).prob_values = prob_values;
    prob=[prob;prob_values];
    figure(1);
    subplot(ceil(nfold/5),5,i);imagesc(CV(i).grid);drawnow;
    
    [CV3d(i).model, CV3d(i).bestParam, CV3d(i).grid ]= learn_on_trainingData(trainData3d, trainLabel, cRange3d, gRange3d, nfoldCV, 1);
  [predict_label3d, accuracy3d, prob_values3d] = svmpredict(testLabel, testData3d, CV3d(i).model,'-b 1');
    acc3d(i).accuracy=accuracy3d(1);
    acc3d(i).testLabel = testLabel;
    acc3d(i).predict_label = predict_label3d;
    acc3d(i).prob_values = prob_values3d;
    prob3d=[prob3d;prob_values3d];
    figure(2);
    subplot(ceil(nfold/5),5,i);imagesc(CV3d(i).grid);drawnow;
    
    disp(['done fold ', num2str(i)]);
end

acc = acc(~isnan(extractfield(acc,'accuracy')));
acc3d = acc3d(~isnan(extractfield(acc3d,'accuracy')));



%% confusion matrix
k=0;
for alfa=0:0.01:1
    k=k+1;
    fused=prob3d*(1-alfa)+prob*alfa;
    [val ind]=max(fused,[],2);
    fusedLabel=ind;
predictLabels = fusedLabel;
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

Precision(k) = mean(diag(ConfusionMatrixPrecision));
Sensitivity(k) = mean(diag(ConfusionMatrixSensitivity));

ave_acc(k)=sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix));
title(['Confusion Matrix, alfa: ' num2str(alfa) ' Acc: ' num2str(100*ave_acc(k)) '% Precision: ' num2str(100*mean(Precision)) '% Recall: ' num2str(100*mean(Sensitivity)) '%']);

end
figure;
plot(0:0.01:1,ave_acc);

saveName=['./EXPproper/' saveName1,saveName2];
saveas(gcf, saveName, 'fig');
save(saveName);
    