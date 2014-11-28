clc
close all
clear all

nfoldCV=3;
%enum{
LAUGHTER = 1;
BREATHING = 1;
OTHER = 1;
REJECT = 2;
%}

% fs = 16000;
% Vfs = 120;
% K=12;
% 
% winms=750; %in ms
% shiftms=250; %frame periodicity in ms
% 
% winSize  = winms/1000*fs;
% winShift = shiftms/1000*fs;
% 
% winSize3d  = winms/1000*Vfs;
% winShift3d = shiftms/1000*Vfs;
% 
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
%         AffectDataSync(end,:).sesNumber = str2num(Samples(j).fileName(5));
%         i  =i + 1;
%         
%     end
%     idcount=idcount+1;
%     disp(['done with the sample ', num2str(j), ' #wins in total: ' num2str(length(AffectDataSync))]);
% end
% 
% 
% save('./Dataset/AffectDataSync+sesNumber', 'AffectDataSync');

load ./Dataset/AffectDataSync+sesNumber

%% CV

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
    datatemp(i,:)=extract_stats(AffectDataSync(i).data);
end

for i=1:length(AffectDataSync)
    data(i,:)=[datatemp(i,:) extract_stats(AffectDataSync(i).data3d)];
end

labelList = unique(label);
NClass = length(labelList);

% 
% 
% %% Leave one Session out test
% 
for k=1:3
    
    testData=data(extractfield(AffectDataSync,'sesNumber')==k,:);
    testLabel=label(extractfield(AffectDataSync,'sesNumber')==k);
    
    trainData=data(extractfield(AffectDataSync,'sesNumber')~=k,:);
    trainLabel=label(extractfield(AffectDataSync,'sesNumber')~=k);
    
    
% % #######################
% % Parameter selection using 3-fold cross validation
% % #######################
bestcv = 0;
i =1; j =1;
for log2c = -2:4:46,
    for log2g = -14:1:-10,
        cmd = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv(i,j) = get_cv_ac_bin(trainLabel, trainData, cmd, nfoldCV);
        if (cv(i,j) >= bestcv),
            bestcv = cv(i,j); bestc = 2^log2c; bestg = 2^log2g;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv(i,j), bestc, bestg, bestcv);
        j = j+1;
    end
    j =1;
    i = i + 1;
end
figure(k);
imagesc(cv);title(['CV for leave session' num2str(k) ' out']);
%% #######################
% % Train the SVM in one-vs-rest (OVR) mode
% % #######################
 bestParam = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg)];

    
    model = svmtrain(trainLabel, trainData, bestParam);
    [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, model);
    
    acc(k).accuracy=accuracy(1);
    acc(k).testLabel = testLabel;
    acc(k).predict_label = predict_label;
    
    disp(['done with ', num2str(k)]);
end

%ave=mean(acc(~isnan(acc)));
acc = acc(~isnan(extractfield(acc,'accuracy')));
%ave = mean(extractfield(acc,'accuracy'));
%fprintf('Ave. Accuracy = %g%%\n', ave);
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
ave=100*sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix));
fprintf('Ave. Accuracy = %g%%\n', ave);

%% Plots!
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

% saveas(gcf, './EXP/DetectionFusedSessionOut', 'fig');
% save ./EXP/DetectionFusedSessionOut