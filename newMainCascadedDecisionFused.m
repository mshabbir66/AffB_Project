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
classifierType=2;


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
NClass = 2;%length(labelList);


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
  
  trainData(trainLabel == REJECT,:) = [];
  trainData3d(trainLabel == REJECT,:) = [];
  trainLabel(trainLabel == REJECT) = [];
  
  
  
  for k=1:length(test_id)
      test_ind=[test_ind;find(extractfield(AffectDataSync,'id')==test_id(k))'];
  end
  testData=data(test_ind,:);
  testData3d=data3d(test_ind,:);
  testLabel=label(test_ind);
  
  %testData(testLabel == REJECT,:) = [];
  %testData3d(testLabel == REJECT,:) = [];
  %testLabel(testLabel == REJECT) = [];
  Mask = testLabel == REJECT;
  
  [CV(i).model, CV(i).bestParam, CV(i).grid ]= learn_on_trainingData(trainData, trainLabel, cRange, gRange, nfoldCV, 1);
  [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, CV(i).model,'-b 1');
    acc(i).accuracy=accuracy(1);
    acc(i).testLabel = testLabel(~Mask);
    acc(i).testLabel2 = testLabel;
    
    acc(i).predict_label = predict_label(~Mask);
    acc(i).predict_label2 = predict_label;
    acc(i).prob_values = prob_values(~Mask,:);
    acc(i).prob_values2 = prob_values;
    %prob=[prob;prob_values];
    figure(1);
    subplot(ceil(nfold/5),5,i);imagesc(CV(i).grid);drawnow;
    
    [CV3d(i).model, CV3d(i).bestParam, CV3d(i).grid ]= learn_on_trainingData(trainData3d, trainLabel, cRange3d, gRange3d, nfoldCV, 1);
  [predict_label3d, accuracy3d, prob_values3d] = svmpredict(testLabel, testData3d, CV3d(i).model,'-b 1');
    acc3d(i).accuracy=accuracy3d(1);
    acc3d(i).testLabel = testLabel(~Mask);
    acc3d(i).predict_label = predict_label3d(~Mask);
    acc3d(i).predict_label2 = predict_label3d;
    
    acc3d(i).prob_values = prob_values3d(~Mask,:);
    acc3d(i).prob_values2 = prob_values3d;
    
    %prob3d=[prob3d;prob_values3d];
    figure(2);
    subplot(ceil(nfold/5),5,i);imagesc(CV3d(i).grid);drawnow;
    
    disp(['done fold ', num2str(i)]);
end

prob3d=[];prob=[];
for i=1:nfold
    prob_values3d_ordered=zeros(size(acc3d(i).prob_values));
    prob_values3d_ordered(:,CV3d(i).model.Label')=acc3d(i).prob_values;
    prob3d=[prob3d;prob_values3d_ordered];
    
    prob_values_ordered=zeros(size(acc(i).prob_values));
    prob_values_ordered(:,CV(i).model.Label)=acc(i).prob_values;
    prob=[prob;prob_values_ordered];
end

acc = acc(~isnan(extractfield(acc,'accuracy')));
acc3d = acc3d(~isnan(extractfield(acc3d,'accuracy')));


%% confusion matrix
ConfusionMatrices(101).confdata=zeros(NClass);
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
    ConfusionMatrices(k).confdata(i,j) = sum(predictLabels(testLabels==i)==j);
    end
end
ConfusionMatrixSensitivity = ConfusionMatrices(k).confdata./(sum(ConfusionMatrices(k).confdata,2)*ones(1,NClass));
ConfusionMatrixPrecision = ConfusionMatrices(k).confdata./(ones(NClass,1)*sum(ConfusionMatrices(k).confdata,1));

%% plot and metrics
% figure;
% bar3(ConfusionMatrix');
% ax = gca;
% set(ax,'XTickLabel',axlabels);
% set(ax,'YTickLabel',axlabels);
% xlabel('GT');
% ylabel('P');

Precision(k) = mean(diag(ConfusionMatrixPrecision));
Sensitivity(k) = mean(diag(ConfusionMatrixSensitivity));

ave_acc(k)=sum(diag(ConfusionMatrices(k).confdata))/sum(sum(ConfusionMatrices(k).confdata));
% title(['Confusion Matrix, alfa: ' num2str(alfa) ' Acc: ' num2str(100*ave_acc(k)) '% Precision: ' num2str(100*mean(Precision)) '% Recall: ' num2str(100*mean(Sensitivity)) '%']);

end
[maxacc, maxind]=max(ave_acc);
figure;
xax=0:0.01:1;
plot(xax,ave_acc);title(['maximum accuracy ' num2str(100*maxacc) '% for alfa='  num2str(xax(maxind))]);

ConfusionMatrix=ConfusionMatrices(maxind).confdata;

%saveName=['./EXPproper/' saveName1,saveName2];
%saveas(gcf, saveName, 'fig');
%save(saveName);

%%
best_alpha = xax(maxind);

prob3d2 = [];
prob2 = [];

for i=1:nfold
    prob_values3d_ordered=zeros(size(acc3d(i).prob_values2));
    prob_values3d_ordered(:,CV3d(i).model.Label')=acc3d(i).prob_values2;
    prob3d2=[prob3d2;prob_values3d_ordered];
    
    prob_values_ordered=zeros(size(acc(i).prob_values2));
    prob_values_ordered(:,CV(i).model.Label)=acc(i).prob_values2;
    prob2=[prob2;prob_values_ordered];
end

fusedRec=prob3d2*(1-best_alpha)+prob2*best_alpha;
[~, labelFusedRec]=max(fusedRec,[],2);

A = load('EXPproper/DetectionDecFused.mat');

fusedDet = A.prob3d*(1-A.xax(A.maxind))+A.prob*A.xax(A.maxind);
[~, labelFusedDet]=max(fusedDet,[],2);
labelFusedDet(labelFusedDet==2) = 3;
labelFusedDet(labelFusedDet==1) = labelFusedRec(labelFusedDet==1);

testLabels = extractfield(acc,'testLabel2');

predictLabels = labelFusedDet';

%testLabels = extractfield(acc3, 'testLabel');
ConfusionMatrix = zeros(3,3);

for i =1:3
    for j = 1:3
    ConfusionMatrix(i,j) = sum(predictLabels(testLabels==i)==j);
    end
end
ConfusionMatrixSensitivity = ConfusionMatrix./(sum(ConfusionMatrix,2)*ones(1,3));
ConfusionMatrixPrecision = ConfusionMatrix./(ones(3,1)*sum(ConfusionMatrix,1));

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




    