clc
close all
clear all

winms=750; %in ms
shiftms=250; %frame periodicity in ms

nfoldCV=3;
fs = 120;
K=12;


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
load ./Dataset/visseq.mat
load PCA


Samples = [AffectBursts;antiAffectBursts(1:round(length(antiAffectBursts)/2))'];

%% Feature Extraction
idcount=1;
AffectData = [];
for j  = 1:length(Samples)
    datamat=zeros(165,size(visseq(j).data{1,3},1));
    for k=1:size(visseq(j).data{1,3},1)
        datamat(:,k)=str2double(strsplit(visseq(j).data{1,3}{k}))';
    end
    i =0;
    while winSize+ winShift*i < size(visseq(j).data{1,3},1)
        PCAcoef = ExtractPCA(datamat(:,1+winShift*i:winSize+winShift*i),U,pcaWmean,K);
        AffectData(end+1,:).data = PCAcoef;%extract_stats(MFCCs);
        AffectData(end,:).id = idcount;
        AffectData(end,:).label = Samples(j).type;
        i  =i + 1;
        
    end
    idcount=idcount+1;
    disp(['done with the sample ', num2str(j)]);
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
    acc(i)=accuracy(1);
    disp(['done with ', num2str(i)]);
end

ave=mean(acc(~isnan(acc)));
fprintf('Ave. Accuracy = %g%%\n', ave);
% 
% 
% %save(['exp_' num2str(winms) '_' num2str(shiftms) '_D'], 'cv', 'acc', 'ave', 'bestParam', 'bestcv', 'nfoldCV' );
