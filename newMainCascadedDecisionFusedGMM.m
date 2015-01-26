clc
close all
clear all

% AffectDataSync = createAffectDataSync;
% save('./Dataset/AffectDataSync+sesNumber', 'AffectDataSync');

load ./Dataset/AffectDataSyncNew

% Removing Other class
AffectDataSync(strcmp(extractfield(AffectDataSync,'label'),'Other'))=[];

nfoldCV = 3;
nfold = 10;


    LAUGHTER = 1;
    BREATHING = 2;
    %OTHER = 1;
    REJECT = 3;
    axlabels={'Laughter','Breathing','Reject'};
    saveName1='Cascaded';


%% label and feature extraction
LABEL=extractfield(AffectDataSync,'label')';
label = zeros(length(LABEL),1);
label(strcmp(LABEL,'Laughter')) = LAUGHTER;
label(strcmp(LABEL,'Breathing')) = BREATHING;
label(strcmp(LABEL,'REJECT')) = REJECT;
%label(strcmp(LABEL,'Other')) = OTHER

labelList = unique(label);
NClass = 2;%length(labelList);
com=8;


saveName2='DecFused';


%% nfold test
folds(nfold).model=[];
folds3d(nfold).model=[];
IDs=unique(extractfield(AffectDataSync,'id'));
len=length(IDs);
load rand_ind.mat%rand_ind = randperm(len);
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
  
  trainData=AffectDataSync(train_ind);
  trainLabel=label(train_ind);
  
  trainData(trainLabel == REJECT) = [];
  trainLabel(trainLabel == REJECT) = [];
  
  testData=AffectDataSync(test_ind);
  testLabel=label(test_ind);
  
  Mask = testLabel == REJECT;
  
  %%%% audio part
  [ model ] =  trainGMM(trainData, trainLabel, com );
    folds(i).model=model;
    Pos=zeros(length(testData),NClass);
    for j=1:length(testData) 
        for class=1:NClass
            [~,Pos(j,class)] = posterior(model(class).obj,testData(j).data);           
        end
    end    
    [v ix] = sort(Pos,2);
    pred=ix(:,1); 
    folds(i).predict_label = pred(~Mask);
    folds(i).predict_label2 = pred;
    folds(i).testLabel = testLabel(~Mask);
    folds(i).testLabel2 = testLabel;
    folds(i).prob_values = Pos(~Mask,:);
    folds(i).prob_values2 = Pos;
  
    %%%%% be careful this is not nice
    for k=1:length(trainData)
        trainData(k).data=trainData(k).data3d;
    end
    for k=1:length(testData)
        testData(k).data=testData(k).data3d;
    end
    %%%%%
    
    %%%% video part
  [ model ] =  trainGMM(trainData, trainLabel, com );
    folds3d(i).model=model;
    Pos=zeros(length(testData),NClass);
    for j=1:length(testData) 
        for class=1:NClass
            [~,Pos(j,class)] = posterior(model(class).obj,testData(j).data);           
        end
    end    
    [v ix] = sort(Pos,2);
    pred=ix(:,1); 
    folds3d(i).predict_label = pred(~Mask);
    folds3d(i).predict_label2 = pred;
    folds3d(i).testLabel = testLabel(~Mask);
    folds3d(i).testLabel2 = testLabel;
    folds3d(i).prob_values = Pos(~Mask,:);
    folds3d(i).prob_values2 = Pos;
    
    disp(['done fold ', num2str(i)]);
end

prob3d=[];prob=[];
for i=1:nfold
    prob3d=[prob3d;folds3d(i).prob_values];
    prob=[prob;folds(i).prob_values];
end



%% confusion matrix
ConfusionMatrices(101).confdata=zeros(NClass);
k=0;
for alfa=0:0.01:1
    k=k+1;
    fused = decisionFuser( prob3d, prob, alfa);
    [val ind]=min(fused,[],2);
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

prob2 = [];prob3d2 = [];

for i=1:nfold
    prob3d2=[prob3d2;folds3d(i).prob_values2];
    prob2=[prob2;folds(i).prob_values2];
end

fusedRec=decisionFuser( prob3d2, prob2, best_alpha);
[~, labelFusedRec]=min(fusedRec,[],2);

A = load('EXPproper/GMMDetectionDecFused.mat');


fusedDet = decisionFuser( A.prob3d, A.prob, A.xax(A.maxind));
[~, labelFusedDet]=min(fusedDet,[],2);
labelFusedDet(labelFusedDet==2) = 3;
labelFusedDet(labelFusedDet==1) = labelFusedRec(labelFusedDet==1);

testLabels = extractfield(folds,'testLabel2');

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

saveName=['./EXPproper/GMM' saveName1,saveName2];
saveas(gcf, saveName, 'fig');
save(saveName);


    