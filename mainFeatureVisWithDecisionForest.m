clc
close all
clear all

nfoldCV=3;
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

% 
% load AffectBurstsSession123Cleaned
% load antiAffectBursts
% load ./Dataset/soundseq.mat
% load ./Dataset/visseq.mat
% load PCA
% 
% Samples = [AffectBursts;antiAffectBursts(1:round(length(antiAffectBursts)/2))'];
% 
% % Feature Extraction
% idcount=1;
% AffectDataSync = [];
% for j  = 1:length(Samples)
%     datamat=zeros(165,size(visseq(j).data{1,3},1));
%     for k=1:size(visseq(j).data{1,3},1)
%         datamat(:,k)=str2double(strsplit(visseq(j).data{1,3}{k}))';
%     end
%     i =0;
%     while winSize3d+ winShift3d*i < size(visseq(j).data{1,3},1)
%         
%         PCAcoef = ExtractPCA(datamat(:,1+winShift3d*i:winSize3d+winShift3d*i),U,pcaWmean,K);
%         AffectDataSync(end+1,:).data3d = PCAcoef;%extract_stats(PCAcoef);
%         
%         MFCCs = ExtractMFCC(soundseq(j).data(1+winShift*i:winSize+winShift*i),fs);
%         AffectDataSync(end,:).data = MFCCs;%extract_stats(MFCCs);
%         
%         AffectDataSync(end,:).id = idcount;
%         AffectDataSync(end,:).label = Samples(j).type;
%         i  =i + 1;
%         
%     end
%     idcount=idcount+1;
%     disp(['done with the sample ', num2str(j), ' #wins in total: ' num2str(length(AffectDataSync))]);
% end
% 
% 
% save('./Dataset/AffectDataSync', 'AffectDataSync');

load ./Dataset/AffectDataSync

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
    data(i,:)=extract_stats(AffectDataSync(i).data3d);
end
% 
% for i=1:length(AffectDataSync)
%     data(i,:)=[datatemp(i,:) extract_stats(AffectDataSync(i).data3d)];
% end

labelList = unique(label);
NClass = length(labelList);
 
% % #######################
% % Parameter selection using 3-fold cross validation
% % #######################

trainData = single(data);
trainLabel = uint8(label);


settings.MaxDecisionLevels = 20;

% Number of candidate feature response functions per split node
% Default: 10.
settings.NumberOfCandidateFeatures = 30;

% Optimal entropy split is determined by thresholding on 
% NumberOfCandidateThresholdsPerFeature equidistant points
% Default 10.
settings.NumberOfCandidateThresholdsPerFeature = 10;

% Number of trees in the forest
% Default: 30.
settings.NumberOfTrees = 30;

% Default: false.
settings.verbose = false;

% Determine which weak learner to be used as a split function.
% Options {random-hyperplane, axis-aligned-hyperplane}
% Default: axis-aligned-hyperplane
settings.WeakLearner = 'random-hyperplane'; 

% Thread(s) used when training and testing.
% Default: 1
settings.MaxThreads =  feature('NumThreads');

% The serialized forest will be saved and loaded from this filename
% Default: forest.bin
settings.forestName = 'temp.bin';

%sherwood_train(trainData', trainLabel, settings);
%probabilities = sherwood_classify(trainData', settings);
%predictLabels = ones(length(probabilities),1);

%predictLabels(probabilities(1,:) < 0.5) = 2;

%disp(num2str(sum((trainLabel==predictLabels))/length(trainLabel)));
%ac = get_cv_ac_bin_decisionBin(trainLabel, trainData, settings, 3);

 bestcv = 0;
 i =1; j =1;
 for MaxDecisionLevels = 5:2:15,
     for NumberOfTrees = 40:20:200
         settings.MaxDecisionLevels = MaxDecisionLevels;
         settings.NumberOfTrees = NumberOfTrees;
         cv(i,j) = get_cv_ac_bin_decisionBin(trainLabel, trainData, settings, 3);
          if (cv(i,j) >= bestcv),
             bestcv = cv(i,j); bestMaxDecisionLevels = MaxDecisionLevels; bestNumberOfTrees = NumberOfTrees;
         end
         fprintf('%g %g %g (best MDL=%g, NOT=%g, rate=%g)\n', MaxDecisionLevels, NumberOfTrees, cv(i,j), bestMaxDecisionLevels, bestNumberOfTrees, bestcv);
         j = j+1;
     end
     j =1;
     i = i + 1;
 end




%% #######################
% % Train the SVM in one-vs-rest (OVR) mode
% % #######################
settings.MaxDecisionLevels = bestMaxDecisionLevels;
settings.NumberOfCandidateFeatures = 30;
settings.NumberOfCandidateThresholdsPerFeature = 10;
settings.NumberOfTrees = bestNumberOfTrees;
settings.verbose = false;
settings.WeakLearner = 'random-hyperplane'; 
settings.MaxThreads =  feature('NumThreads');
settings.forestName = 'temp_leaveOneOut.bin';


for i=1:max(extractfield(AffectDataSync,'id'))
    testData=single(data(extractfield(AffectDataSync,'id')==i,:));
    testLabel=uint8(label(extractfield(AffectDataSync,'id')==i));
    
    trainData=single(data(extractfield(AffectDataSync,'id')~=i,:));
    trainLabel=uint8(label(extractfield(AffectDataSync,'id')~=i));
    
    sherwood_train(trainData', trainLabel, settings);
  
    probabilities = sherwood_classify(testData', settings);
    [~,predict_label] = max(probabilities,[],1);
    predict_label = predict_label';

    
    
%     testData=data(extractfield(AffectDataSync,'id')==i,:);
%     testLabel=label(extractfield(AffectDataSync,'id')==i);
%     
%     trainData=data(extractfield(AffectDataSync,'id')~=i,:);
%     trainLabel=label(extractfield(AffectDataSync,'id')~=i);
%     
%     model = svmtrain(trainLabel, trainData, bestParam);
%     [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, model);
%     
    acc(i).accuracy= sum((testLabel==predict_label))/length(testLabel);
    acc(i).testLabel = testLabel;
    acc(i).predict_label = predict_label;
    acc(i).prob = probabilities;
    
    disp(['done with ', num2str(i)]);
end

%%
%ave=mean(acc(~isnan(acc)));
acc = acc(~isnan(extractfield(acc,'accuracy')));
ave = mean(extractfield(acc,'accuracy'));
fprintf('Ave. Accuracy = %g%%\n', ave);
predictLabels = extractfield(acc, 'predict_label');
testLabels = extractfield(acc, 'testLabel');
for i =1:NClass
    for j = 1:NClass
    ConfusionMatrix(i,j) = sum(predictLabels(testLabels==i)==j);
    end
end
ConfusionMatrixSensitivity = ConfusionMatrix./(sum(ConfusionMatrix,2)*ones(1,NClass));
ConfusionMatrixPrecision = ConfusionMatrix./(ones(NClass,1)*sum(ConfusionMatrix,1));

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

saveas(gcf, './EXP/RecognitionVis_3class_decision', 'fig');
save ./EXP/RecognitionVis_3class_decision