clc
close all
clear all


fs = 16000;
Vfs = 120;
K=12;
load PCA_ses1234.mat

winms=750; %in ms
shiftms=250; %frame periodicity in ms

winSize  = winms/1000*fs;
winShift = shiftms/1000*fs;

winSize3d  = winms/1000*Vfs;
winShift3d = shiftms/1000*Vfs;

fileName = 'Ses04F_impro03';
[y,fs] = wavread(['..\Session',fileName(5),'\dialog\wav\',fileName,'.wav']);
%[y,fs] = wavread('..\Session4\dialog\wav\Ses04F_impro03.wav');
%y=y(:,2)';

switch str2num(fileName(5))
    case 1
        y = y(startFrame:endFrame,1);
    case 2
        y = y(startFrame:endFrame,2);
    case 3
        y = y(startFrame:endFrame,2);
    case 4
        y = y(startFrame:endFrame,2);
end

fidv = fopen(['../Session',fileName(5),'/dialog/MOCAP_rotated/',fileName ,'.txt'],'r');
text=textscan(fidv,'%d %f %s','Delimiter','\n','Headerlines',2);
fclose(fidv);

datamat=zeros(165,size(text{1,3},1));
for k=1:size(text{1,3},1)
    datamat(:,k)=str2double(strsplit(text{1,3}{k}))';
end

numberOfFrames=length(y)*1000/fs;
unseenStats = [];
i=0;
while winSize+ winShift*i < length(y)%winSize3d+ winShift3d*i < size(text{1,3},1)
    PCAcoef = ExtractPCA(datamat(:,1+winShift3d*i:winSize3d+winShift3d*i),U,pcaWmean,K);
    MFCCs = ExtractMFCC(y(1+winShift*i:winSize+ winShift*i),fs);
    unseenStats(end+1,:) =[extract_stats(MFCCs) extract_stats(PCAcoef)] ;
    i = i + 1;
end

%%

load ./Dataset/AffectDataSyncSes4Injected

% Removing Other class
AffectDataSync(strcmp(extractfield(AffectDataSync,'label'),'Other'))=[];
AffectDataSync(strcmp(extractfield(AffectDataSync,'fileName'),fileName))=[];

nfoldCV = 3;
nfold = 10;

% detection 1, recognition 2
classifierType=1;

if(classifierType==1)
    LAUGHTER = 1;
    BREATHING = 1;
    OTHER = 2;
    REJECT = 2;
    axlabels={'Affect Burst','Reject'};
    saveName1='Detection';
else
    LAUGHTER = 1;
    BREATHING = 2;
    OTHER = 3;
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


for i=1:length(AffectDataSync)
    datatemp(i,:)=extract_stats(AffectDataSync(i).data);
    data(i,:)=[datatemp(i,:) extract_stats(AffectDataSync(i).data3d)];
end
cRange=[-2 4 46];
gRange=[-13 1 -10];
saveName2='Fused';


[model, bestParam, grid ]= learn_on_trainingData(data, label, cRange, gRange, nfoldCV, 0);

[predict_label, accuracy, prob_values] = svmpredict(zeros(size(unseenStats,1),1), unseenStats, model);



%% ground truth compare
labelmap = containers.Map;
labelmap('Laughter') = LAUGHTER;
labelmap('Breathing') = BREATHING;
labelmap('Other') = OTHER;
labelmap('REJECT') = REJECT;

Aff=load('AffectBurstsSession1234Cleaned');
AffectDataSync(strcmp(extractfield(AffectBursts,'fileName'),fileName))=[];

% load(['./Dataset/' fileName]);
real_label = GenerateAffectBurstLabelsForSingleFile(Ses04,fileName,numberOfFrames,labelmap);

t=0:1/1000:length(real_label)/1000-1/1000;
twin=0:shiftms/1000:length(real_label)/1000-shiftms/1000;
real_label_scaled = zeros(size(predict_label));
predict_label_r_d = zeros(size(predict_label));

twin=twin(2:end-1);
for i =2:length(twin)
    real_label_scaled(i) = median(double(real_label((t < twin(i)) & (t > twin(i-1)))));
end

real_label_scaled(real_label_scaled==0) = REJECT;


for i =6:length(twin)-5
    predict_label_r_d(i) = median(predict_label(i-5:i+5,1));
end

disp('Frame Wise calculations');
disp(['Acc = ', num2str(sum(real_label_scaled == (predict_label_r_d))/length(predict_label_r_d))]);
disp(['FP= ', num2str(sum(real_label_scaled==2 & (predict_label_r_d==1)))]);
disp(['TP= ', num2str(sum(real_label_scaled==1 & (predict_label_r_d==1)))]);
disp(['TN= ', num2str(sum(real_label_scaled==2 & (predict_label_r_d==2)))]);
disp(['FN= ', num2str(sum(real_label_scaled==1 & (predict_label_r_d==2)))]);
Acc = sum(real_label_scaled == (predict_label_r_d))/length(predict_label_r_d);
FP = sum(real_label_scaled==2 & (predict_label_r_d==1));
TP = sum(real_label_scaled==1 & (predict_label_r_d==1));
TN = sum(real_label_scaled==2 & (predict_label_r_d==2));
FN = sum(real_label_scaled==1 & (predict_label_r_d==2));

ave_acc=100*(TP+TN)/(TP+FP+TN+FN);
precision=100*TP/(TP+FP);
recall=100*TP/(TP+FN);

%% Plots
subplot(2,1,1);
bar(twin,predict_label_r_d==LAUGHTER,'b','EdgeColor','None');
hold on;
bar(twin,predict_label_r_d==BREATHING,'r','EdgeColor','None');

title('Predicted');
%line([step,step],[0,1],'LineWidth',2,'Color','g');
hold off;


subplot(2,1,2);
bar(twin,real_label_scaled==LAUGHTER,'b','EdgeColor','None');
hold on
bar(twin,real_label_scaled==BREATHING,'r','EdgeColor','None');
title('Real');
%line([step,step],[0,1],'LineWidth',2,'Color','g');
hold off;
drawnow;


%saveas(gcf, './EXP/DetectionFused_Unseen', 'fig');
%save ./EXP/DetectionFused_Unseen

