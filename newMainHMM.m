%function [predictLabels] = newMainHMM(noS,noM,classifierType)

load ./Dataset/AffectDataSyncN
% Removing Other class
AffectDataSync(strcmp(extractfield(AffectDataSync,'label'),'Other'))=[];

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
CV3D(nfold).model = [];

accCombined(nfold).predict_label =[];

acc(nfold).accuracy = [];
acc(nfold).testLabel = [];
acc(nfold).predict_label = [];
acc(nfold).prob_values = [];

acc3D(nfold).accuracy = [];
acc3D(nfold).testLabel = [];
acc3D(nfold).predict_label = [];
acc3D(nfold).prob_values = [];

IDs=unique(extractfield(AffectDataSync,'id'));
len=length(IDs);
load rand_ind.mat%rand_ind = randperm(len);
rand_id = IDs(rand_ind);

alfa=0.5;
%noS  = 4;
%noM = 4;

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
       
    [CV(i).model ]= trainHMMGMM(trainDataSound, trainLabel,noS,noM);
    
    [predict_label,~, prob_values] = testHMMGMM(testLabel, testDataSound, CV(i).model);
    
    acc(i).accuracy=sum(testLabel==predict_label)/length(predict_label);
    acc(i).testLabel = testLabel;
    acc(i).predict_label = predict_label;
    acc(i).prob_values = prob_values;

    [CV3D(i).model ]= trainHMMGMM(trainData3D, trainLabel,noS,noM);
    
    [predict_label,~, prob_values] = testHMMGMM(testLabel, testData3D, CV3D(i).model);
    
    acc3D(i).accuracy=sum(testLabel==predict_label)/length(predict_label);
    acc3D(i).testLabel = testLabel;
    acc3D(i).predict_label = predict_label;
    acc3D(i).prob_values = prob_values;
    %subplot(ceil(nfold/5),5,i);imagesc(CV(i).grid);drawnow;
        
    [ fusedLabel] = decisionFuserModified( acc3D(i).prob_values,acc(i).prob_values, 0.5);
    [val, ind]=max(fusedLabel,[],2);
    
    accCombined(i).predict_label = ind;
    
    disp(['done fold ', num2str(i)]);
end


%% confusion matrix

ConfusionMatrices(101).confdata=zeros(NClass);
k=0;

prob3d=[];prob=[];
for i=1:nfold
    prob3d=[prob3d;acc3D(i).prob_values];
    prob=[prob;acc(i).prob_values];
end

for alfa=0:0.01:1
    k=k+1;
    fused = decisionFuserModified( prob, prob3d, alfa);
    
    [val, ind]=max(fused,[],2);
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
best_alpha = xax(maxind);

predictLabels = extractfield(accCombined, 'predict_label');
testLabels = extractfield(acc, 'testLabel');

ConfusionMatrix(NClass,NClass) = 0;

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

 saveName=['./EXPproper/HMMGMMDetectionDecFused-',num2str(noS),'-',num2str(noM)];
 
 saveas(gcf, saveName, 'fig');
 save(saveName);
