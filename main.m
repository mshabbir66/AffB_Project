clc
close all
clear all

winms=750; %in ms
shiftms=250; %frame periodicity in ms

nfoldCV=3;
fs = 16000;

winSize  = winms/1000*fs;
winShift = shiftms/1000*fs;

%enum{
LAUGHTER = 1;
BREATHING = 2;
OTHER = 3;
REJECT = 4;
%}

load AffectBurstsSession123Cleaned
load antiAffectBursts
load ./Dataset/soundseq.mat


Samples = [AffectBursts;antiAffectBursts(1:round(length(antiAffectBursts)/2))'];

%% Feature Extraction
idcount=1;
AffectData = [];
for j  = 1:length(Samples)
    i =0;
    while winSize+ winShift*i < length(soundseq(j).data)
        MFCCs = ExtractMFCC(soundseq(j).data(1+winShift*i:winSize+winShift*i),fs);
        AffectData(end+1,:).data = MFCCs;%extract_stats(MFCCs);
        AffectData(end,:).id = idcount;
        AffectData(end,:).label = Samples(j).type;
        i  =i + 1;
        
    end
    idcount=idcount+1;
end


save ./Dataset/AffectData AffectData

%load ./Dataset/AffectData

%% CV

addpath C:\Users\Shabbir\Desktop\libsvm-3.18\libsvm-3.18\matlab
ind = randperm(length(AffectData))';
AffectData = AffectData(ind,:);
 
LABEL=extractfield(AffectData,'label')';
label = zeros(length(LABEL),1);
label(strcmp(LABEL,'Laughter')) = LAUGHTER;
label(strcmp(LABEL,'Breathing')) = BREATHING;
label(strcmp(LABEL,'Other')) = OTHER;
label(strcmp(LABEL,'REJECT')) = REJECT;

%data=zeros(length(AffectData),length(AffectData(1).data));

for i=1:length(AffectData)
    data(i,:)=extract_stats(AffectData(i).data);
end

labelList = unique(label);
NClass = length(labelList);
 
% % #######################
% % Parameter selection using 3-fold cross validation
% % #######################
bestcv = 0;
i =1; j =1;
for log2c = -2:4:34,
    for log2g = -13:1:-7,
        cmd = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv(i,j) = get_cv_ac_bin(label, data, cmd, nfoldCV);
        if (cv(i,j) >= bestcv),
            bestcv = cv(i,j); bestc = 2^log2c; bestg = 2^log2g;
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv(i,j), bestc, bestg, bestcv);
        j = j+1;
    end
    j =1;
    i = i + 1;
end

%% #######################
% % Train the SVM in one-vs-rest (OVR) mode
% % #######################
 bestParam = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg)];


% 
% 
% %% Leave one out test
% 
parfor i=1:max(extractfield(AffectData,'id'))
    testData=data(extractfield(AffectData,'id')==i,:);
    testLabel=label(extractfield(AffectData,'id')==i);
    
    trainData=data(extractfield(AffectData,'id')~=i,:);
    trainLabel=label(extractfield(AffectData,'id')~=i);
    
    model = svmtrain(trainLabel, trainData, bestParam);
    [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, model);
    
    acc(i).accuracy=accuracy(1);
    acc(i).testLabel = testLabel;
    acc(i).predict_label = predict_label;
    
    disp(['done with ', num2str(i)]);
end

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


saveas(gcf, './EXP/Recognition', 'fig');
