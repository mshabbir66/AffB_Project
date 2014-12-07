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


[y,fs] = wavread('..\Session4\dialog\wav\Ses04F_impro03.wav');
y=y(:,2)';

fidv = fopen('../Session4/dialog/MOCAP_rotated/Ses04F_impro03.txt','r');
text=textscan(fidv,'%d %f %s','Delimiter','\n','Headerlines',2);
fclose(fidv);

datamat=zeros(165,size(text{1,3},1));
    for k=1:size(text{1,3},1)
        datamat(:,k)=str2double(strsplit(text{1,3}{k}))';
    end

%load ./Dataset/Ses04F_impro03Textdatamat

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
load('./EXPproper/newTest.mat');%,'cRange', 'gRange', 'nfoldCV', 'data', 'label');
[model, bestParam, grid ]= learn_on_trainingData(data, label, cRange, gRange, nfoldCV, 0);

[predict_label, accuracy, prob_values] = svmpredict(zeros(size(unseenStats,1),1), unseenStats, model);

%% ground truth compare
labelmap = containers.Map;
labelmap('Laughter') = LAUGHTER;
labelmap('Breathing') = BREATHING;
labelmap('Other') = OTHER;
labelmap('REJECT') = REJECT;

file = 'Ses04F_impro03';
suffix = '';
temp=[file suffix];
load(['./Dataset/' file suffix]);
real_label = GenerateAffectBurstLabelsForSingleFile(Ses04,temp,numberOfFrames,labelmap);


figure(1);
t=0:1/1000:length(real_label)/1000-1/1000;
twin=0:shiftms/1000:length(real_label)/1000-shiftms/1000;
real_label_scaled = zeros(size(predict_label));
predict_label_r_d = zeros(size(predict_label));


twin=twin(2:end-1);
for i =2:length(twin)
    real_label_scaled(i) = median(double(real_label((t < twin(i)) & (t > twin(i-1)))));
end

real_label_scaled(real_label_scaled==0) = REJECT;


subplot(2,1,1);
    
    bar(twin,predict_label==LAUGHTER,'b','EdgeColor','None');
    hold on;
    bar(twin,predict_label==BREATHING,'r','EdgeColor','None');
    
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

