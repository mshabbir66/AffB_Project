clc
close all
clear all

% AffectDataSync = createAffectDataSync;
% save('./Dataset/AffectDataSync+sesNumber', 'AffectDataSync');

load ./Dataset/AffectDataSyncN

% Removing Other class
AffectDataSync(strcmp(extractfield(AffectDataSync,'label'),'Other'))=[];

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
%label(strcmp(LABEL,'Other')) = OTHER

labelList = unique(label);
NClass = length(labelList);



%% nfold test
CV(nfold).model=[];
IDs=unique(extractfield(AffectDataSync,'id'));
len=length(IDs);
load rand_ind.mat%rand_ind = randperm(len);
rand_id = IDs(rand_ind);
alfa=0.5;


for i=1:nfold % nfold test
    train_ind=[];test_ind=[];
    test_id=rand_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]');
    train_id = rand_id;
    train_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]) = [];
    
    for k=1:length(train_id)
        train_ind=[train_ind;find(extractfield(AffectDataSync,'id')==train_id(k))'];
    end
    
    trainData=AffectDataSync(train_ind,:);
    trainLabel=label(train_ind);
    
    trainDataCell = struct2cell(trainData);
    %select sound
    trainDataSound = trainDataCell(2,:)';
    trainData3D = trainDataCell(1,:)';
    
    for k = 1:length(trainData)
        trainDataSound{k} = trainDataSound{k}';
        trainData3D{k} = trainData3D{k}';
    end
    
    for k=1:length(test_id)
        test_ind=[test_ind;find(extractfield(AffectDataSync,'id')==test_id(k))'];
    end
    
    testData=AffectDataSync(test_ind,:);
    testLabel=label(test_ind);
    
    testDataCell = struct2cell(testData);
    %select sound
    testDataSound = testDataCell(2,:)';
    testData3D = testDataCell(1,:)';
    
    for k = 1:length(testData)
        testDataSound{k} = testDataSound{k}';
        testData3D{k} = testData3D{k}';
    end
       
    %[CV(i).model ]= trainHMMGMM(trainDataSound, trainLabel);
    [ CV(i).model, bestComS, cvS ] = CVtrainHMM(trainData,trainDataSound, trainLabel, 1:1:4, nfoldCV);
    
    [predict_label,~, prob_values] = testHMMGMM(testLabel, testDataSound, CV(i).model);
    
    

    acc(i).accuracy=sum(testLabel==predict_label)/length(predict_label);
    acc(i).testLabel = testLabel;
    acc(i).predict_label = predict_label;
    acc(i).prob_values = prob_values;

    [CV3D(i).model, CV3D(i).bestCom, CV3D(i).cv ] = CVtrainHMM(trainData,trainData3D, trainLabel, 1:1:4, nfoldCV);
    %[CV(i).model ]= trainHMMGMM(trainData3D, trainLabel);
    
    [predict_label,~, prob_values] = testHMMGMM(testLabel, testData3D, CV3D(i).model);
    
    acc3d(i).accuracy=sum(testLabel==predict_label)/length(predict_label);
    acc3d(i).testLabel = testLabel;
    acc3d(i).predict_label = predict_label;
    acc3d(i).prob_values = prob_values;
    %subplot(ceil(nfold/5),5,i);imagesc(CV(i).grid);drawnow;
        
    [ fusedLabel2] = decisionFuser( acc3d(i).prob_values,acc(i).prob_values, 0.5);
    [val ind]=max(fusedLabel2,[],2);
    
    accCombined(i).predict_label = ind;
    
    disp(['done fold ', num2str(i)]);
end


%% confusion matrix
predictLabels = extractfield(accCombined, 'predict_label');
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

% saveName=['./EXPproper/' saveName1,saveName2];
% saveas(gcf, saveName, 'fig');
% save(saveName);
