clc
close all
clear all


load stats_750_250_0_D
% load exp file here
winms = 750;
shiftms = 250;
fs= 16000;
winSize  = winms/1000*fs;
winShift = shiftms/1000*fs;


ind = randperm(length(Stats))';
Stats = Stats(ind,:);

trainLabel=extractfield(Stats,'label')';
trainData=zeros(length(Stats),length(Stats(1).data));
for i=1:length(Stats)
    data(i,:)=extractfield(Stats(i),'data');
end

[y,fs] = wavread('F:\Datasets\IEMOCAP\Session4\dialog\wav\Ses04F_impro03.wav');
y=y(:,2)';

numberOfFrames=length(y)*1000/fs;
unseenStats = [];
i=0;
while winSize+ winShift*i < length(y)
        MFCCs = ExtractMFCC(y(1+winShift*i:winSize+ winShift*i),fs);
        unseenStats(end+1,:) = extract_stats(MFCCs);
        i  =i + 1;      
end
    
%%
load decision_best_Params_750_250_0_D

% Best Params

settings.MaxDecisionLevels = bestMaxDecisionLevels;

% Number of candidate feature response functions per split node
% Default: 10.
settings.NumberOfCandidateFeatures = 10;

% Optimal entropy split is determined by thresholding on 
% NumberOfCandidateThresholdsPerFeature equidistant points
% Default 10.
settings.NumberOfCandidateThresholdsPerFeature = 10;

% Number of trees in the forest
% Default: 30.
settings.NumberOfTrees = bestNumberOfTrees;

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
settings.forestName = 'temp_test_on_seq.bin';

model = svmtrain(label, data, bestParam);
[predict_label, accuracy, prob_values] = svmpredict(zeros(size(unseenStats,1),1), unseenStats, model);
predict_label = 2-predict_label;

%%
load('Ses04F_impro03.mat');
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
fp.starttimes = twin(conv(double(false_positives),[-1 1],'same')<0)*winms/1000;
fp.endtimes = twin(conv(double(false_positives),[-1 1],'same')>0)*winms/1000;
%fp.endtimes =  fp.endtimes(2:end); % first sample bogus
offset = 0.1;%in sec
for i = 1:length(fp.starttimes)

    if ((fp.starttimes(i) - offset >0) && (fp.endtimes(i) + offset < length(y)/fs))
        fpsig = y(round((fp.starttimes(i) - offset) * fs) : round((fp.endtimes(i) + offset) * fs)); 
        sound(fpsig,fs);
    end
    pause
end 
