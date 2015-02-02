clear all;
clc
close all

% AffectDataSync = createAffectDataSync;
% save('./Dataset/AffectDataSync+sesNumber', 'AffectDataSync');

load ./Dataset/AffectDataSyncNew

% Removing Other class
AffectDataSync(strcmp(extractfield(AffectDataSync,'label'),'Other'))=[];

nfold = 10;


% detection 1, recognition 2
classifierType=2;

% audio 1, video 2, feature fusion 3


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
com=2;


%% nfold test
folds(nfold).modelVideo=[];
folds(nfold).modelSound=[];

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
    trainData=AffectDataSync(train_ind,:);
    trainLabel=label(train_ind);
    
    for k=1:length(test_id)
        test_ind=[test_ind;find(extractfield(AffectDataSync,'id')==test_id(k))'];
    end
    testData=AffectDataSync(test_ind,:);
    testLabel=label(test_ind);
    
    %[ model ] =  trainGMM(trainData, trainLabel, com );
    [ modelSound ] = TrainGMMAudio( trainData, trainLabel, com );
    [ modelVideo ] = TrainGMMVideo( trainData, trainLabel, com );
    folds(i).modelSound=modelSound;
    folds(i).modelVideo=modelVideo;
    
    [prob,~] = TestGMMAudio(testData,modelSound,NClass);
    [prob3d,~] = TestGMMVideo(testData,modelVideo,NClass);
    fused = decisionFuser( prob3d, prob, 0.5);
    [val ind]=min(fused,[],2);
    predict_label=ind;
    
    %acc=sum(ix(:,1)==testLabel)/length(testLabel);
    folds(i).testLabel = testLabel;
    folds(i).predict_label = predict_label;
    folds(i).prob_values_audio = prob;
    folds(i).prob_values_video = prob3d;
    disp(['done fold ', num2str(i)]);
end

prob3d=[];prob=[];
for i=1:nfold
    prob3d=[prob3d;folds(i).prob_values_video];
    prob=[prob;folds(i).prob_values_audio];
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
    testLabels = extractfield(folds, 'testLabel');
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


% %% confusion matrix
% predictLabels = extractfield(folds, 'predict_label');
% testLabels = extractfield(folds, 'testLabel');
% for i =1:NClass
%     for j = 1:NClass
%         ConfusionMatrix(i,j) = sum(predictLabels(testLabels==i)==j);
%     end
% end
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

