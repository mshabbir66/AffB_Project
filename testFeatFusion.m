clc
close all
clear all

fs = 16000;
Vfs = 120;
K=12;
load PCA

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

%load ./EXP/Ses04F_impro03Textdatamat

numberOfFrames=length(y)*1000/fs;
unseenStats = [];
i=0;
while winSize+ winShift*i < length(y)%winSize3d+ winShift3d*i < size(text{1,3},1)
    PCAcoef = ExtractPCA(datamat(:,1+winShift3d*i:winSize3d+winShift3d*i),U,pcaWmean,K);
    MFCCs = ExtractMFCC(y(1+winShift*i:winSize+ winShift*i),fs);
    unseenStats(end+1,:) = [extract_stats(MFCCs) extract_stats(PCAcoef)] ;
    i = i + 1;      
end

%%
load('./EXP/DetectionFused_1.mat','bestParam', 'data', 'label');
%bestParam = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg)];

model = svmtrain(label, data, bestParam);
[predict_label, accuracy, prob_values] = svmpredict(zeros(size(unseenStats,1),1), unseenStats, model);
predict_label = 2-predict_label;

%%
load('./Dataset/Ses04F_impro03.mat');
real_label = GenerateAffectBurstLabelsForSingleFile(Ses04,'Ses04F_impro03',numberOfFrames);



figure(1);
t=0:1/1000:length(real_label)/1000-1/1000;
twin=0:shiftms/1000:length(real_label)/1000-shiftms/1000;
real_label_scaled = zeros(size(predict_label));
predict_label_r_d = zeros(size(predict_label));


twin=twin(2:end-1);
for i =2:length(twin)
real_label_scaled(i) = median(double(real_label((t < twin(i)) & (t > twin(i-1)))));
end

for i =3:length(twin)-2
    if (median(predict_label(i-2:i+2,1)) == 1)    
        predict_label_r_d(i) = 1;
        
    end
end
temp = predict_label_r_d;
predict_label_r_d = zeros(size(predict_label));
for i =3:length(twin)-2
    if temp(i) == 1    
      predict_label_r_d(i-2:i+2,1) = ones(5,1);
        
    end
end

false_positives  = (~real_label_scaled) & (predict_label_r_d);

ax=[];

subplot(2,1,1);title('predicted');
bar(twin,predict_label_r_d)
hold on;
bar(twin,false_positives,'r')

axis ([0 300 0 1])
ax(end+1)=gca;
subplot(2,1,2);title('real');
%bar(t,real_label)
bar(twin,real_label_scaled,'b');

axis ([0 300 0 1])
ax(end+1)=gca;
linkaxes(ax,'x'); pan xon; zoom xon;

disp('Frame Wise calculations');
disp(['Acc = ', num2str(sum(real_label_scaled == (predict_label_r_d))/length(predict_label_r_d))]);
disp(['FP= ', num2str(sum(~real_label_scaled & (predict_label_r_d)))]);
disp(['TP= ', num2str(sum(real_label_scaled & (predict_label_r_d)))]);
disp(['TN= ', num2str(sum(~real_label_scaled & ~(predict_label_r_d)))]);
disp(['FN= ', num2str(sum(real_label_scaled & ~(predict_label_r_d)))]);

%%
% fp.starttimes = twin(conv(double(false_positives),[-1 1],'same')<0)*winms/1000;
% fp.endtimes = twin(conv(double(false_positives),[-1 1],'same')>0)*winms/1000;
% %fp.endtimes =  fp.endtimes(2:end); % first sample bogus
% offset = 0.1;%in sec
% for i = 1:length(fp.starttimes)
% 
%     if ((fp.starttimes(i) - offset >0) && (fp.endtimes(i) + offset < length(y)/fs))
%         fpsig = y(round((fp.starttimes(i) - offset) * fs) : round((fp.endtimes(i) + offset) * fs)); 
%         sound(fpsig,fs);
%     end
%     pause
% end 
