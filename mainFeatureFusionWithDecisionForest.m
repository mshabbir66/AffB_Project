clc
close all
clear all

%enum{
LAUGHTER = 1;
BREATHING = 2;
OTHER = 3;
REJECT = 3;

%}

winms=750; %in ms
shiftms=250; %frame periodicity in ms
fs = 16000;
Vfs = 120;
K=12;

winSize  = winms/1000*fs;
winShift = shiftms/1000*fs;

winSize3d  = winms/1000*Vfs;
winShift3d = shiftms/1000*Vfs;

load ./Dataset/AffectDataSyncP

%% CV
addpath('./sherwood-classify-matlab')
%addpath C:\Users\Shabbir\Desktop\libsvm-3.18\libsvm-3.18\matlab
ind = randperm(length(AffectDataSync))';
AffectDataSync = AffectDataSync(ind,:);

LABEL=extractfield(AffectDataSync,'label')';
label = zeros(length(LABEL),1);
label(strcmp(LABEL,'Laughter')) = LAUGHTER;
label(strcmp(LABEL,'Breathing')) = BREATHING;
label(strcmp(LABEL,'Other')) = OTHER;
label(strcmp(LABEL,'REJECT')) = REJECT;

%data=zeros(length(AffectData),length(AffectData(1).data));

for i=1:length(AffectDataSync)
    data(i,:)=[extract_stats(AffectDataSync(i).data),extract_stats(AffectDataSync(i).data3d)];
end

labelList = unique(label);
NClass = length(labelList);

trainData = single(data);
trainLabel = uint8(label);



%% #######################
% % Train the SVM in one-vs-rest (OVR) mode
% % #######################
% settings.MaxDecisionLevels = bestMaxDecisionLevels;
% settings.NumberOfCandidateFeatures = 30;
% settings.NumberOfCandidateThresholdsPerFeature = 10;
% settings.NumberOfTrees = bestNumberOfTrees;
% settings.verbose = false;
% settings.WeakLearner = 'random-hyperplane';
% settings.MaxThreads =  feature('NumThreads');
% settings.forestName = 'temp_leaveOneOut.bin';


nfold = 10;

CV(nfold).model=[];
IDs=unique(extractfield(AffectDataSync,'id'));
len=length(IDs);
load rand_ind.mat
%rand_ind = randperm(len);
rand_id = IDs(rand_ind);
figure;
for i=1:nfold % nfold test
    train_ind=[];test_ind=[];
    test_id=rand_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]');
    train_id = rand_id;
    train_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]) = [];
    
    for k=1:length(train_id)
        train_ind=[train_ind;find(extractfield(AffectDataSync,'id')==train_id(k))'];
    end
    trainData=data(train_ind,:);
    trainLabel=label(train_ind);
    
    for k=1:length(test_id)
        test_ind=[test_ind;find(extractfield(AffectDataSync,'id')==test_id(k))'];
    end
    testData=data(test_ind,:);
    testLabel=label(test_ind);

    %sherwood_train(trainData', trainLabel, settings);
    [settings, cv ] = learn_on_trainingData_Dec(trainData, trainLabel);
    
    probabilities = sherwood_classify(testData', settings);
    [~,predict_label] = max(probabilities,[],1);
    predict_label = predict_label';
    
    
    acc(i).accuracy= sum((testLabel==predict_label))/length(testLabel);
    acc(i).testLabel = testLabel;
    acc(i).predict_label = predict_label;
    acc(i).prob_values = probabilities;
    
    %subplot(ceil(nfold/5),5,i);imagesc(CV(i).grid);drawnow;
    
    disp(['done fold ', num2str(i)]);
end

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

%%
%
%
% %save(['exp_' num2str(winms) '_' num2str(shiftms) '_D'], 'cv', 'acc', 'ave', 'bestParam', 'bestcv', 'nfoldCV' );

% Plots!
figure;
set(gcf,'Position',[50 50 1200 600]);

subplot(1,3,1)
bar3(ConfusionMatrix);
title('Confusion Matrix')
xlabel('GT');
ylabel('P')
subplot(1,3,2)
bar3(ConfusionMatrixSensitivity);
title('Confusion Matrix(Sensitivity)')
xlabel('GT');
ylabel('P');
subplot(1,3,3)

bar3(ConfusionMatrixPrecision);
title('Confusion Matrix(Precision)')
xlabel('GT');
ylabel('P');


% saveas(gcf, './EXP/DetectionFused_1', 'fig');
% save ./EXP/DetectionFused_1

% saveas(gcf, './EXP/RecognitionFused_1', 'fig');
% save ./EXP/RecognitionFused_1

%saveas(gcf, './EXP/RecognitionFused_3class_decision', 'fig');
%save ./EXP/RecognitionFused_3class_decision