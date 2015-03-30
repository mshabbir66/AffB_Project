function [predict_label_r_d, real_label_scaled]=test_on_french(audioPath,EmotionEvents,featurePath,filterSizePred)

LAUGHTER = 1;
BREATHING = 2;
REJECT = 3;

file=audioPath(end-21:end-4);
%% training a model

% load ./Dataset/AffectDataSync
% % Removing Other class
% AffectDataSync(strcmp(extractfield(AffectDataSync,'label'),'Other'))=[];
% 
% % label and feature extraction
% LABEL=extractfield(AffectDataSync,'label')';
% label = zeros(length(LABEL),1);
% label(strcmp(LABEL,'Laughter')) = LAUGHTER;
% label(strcmp(LABEL,'Breathing')) = BREATHING;
% label(strcmp(LABEL,'REJECT')) = REJECT;
% 
% 
% labelList = unique(label);
% NClass = length(labelList);
% datatemp=[]; data=[];
% for i=1:length(AffectDataSync)
%     data(i,:)=extract_stats(AffectDataSync(i).data);
% end
% 
% nfoldCV = 3;
% cRange=[-2 4 34];
% gRange=[-13 1 -8];
% 
% [model, bestParam, grid ]= learn_on_trainingData(data, label, cRange, gRange, nfoldCV, 0);

load('Misc\modelOfWholeSet.mat');

%% testing a sequence

[y,fs] = wavread(audioPath);
winms = 750;
shiftms = 250;
winSize  = winms/1000*fs;
winShift = shiftms/1000*fs;


numberOfFrames=length(y)*1000/fs;
unseenStats = [];
i=0;
while winSize+ winShift*i < length(y)
        MFCCs = ExtractMFCC(y(1+winShift*i:winSize+ winShift*i),fs);
        unseenStats(end+1,:) = extract_stats(MFCCs);
        i  =i + 1;      
end

[predict_label, ~ , ~] = svmpredict(zeros(size(unseenStats,1),1), unseenStats, model);


mask = strcmp(extractfield(EmotionEvents,'type'),'Laughter') | strcmp(extractfield(EmotionEvents,'type'),'Breathing');
EmotionEvents=EmotionEvents(mask);

labelmap = containers.Map;
labelmap('Laughter') = LAUGHTER;
labelmap('Breathing') = BREATHING;
labelmap('REJECT') = REJECT;

real_label = GenerateAffectBurstLabelsForSingleFile(EmotionEvents,file,numberOfFrames,labelmap);


t=0:1/1000:length(real_label)/1000-1/1000;
twin=0:shiftms/1000:length(real_label)/1000-shiftms/1000;
real_label_scaled = zeros(size(predict_label));
predict_label_r_d = zeros(size(predict_label));

twin=twin(2:end-1);
for i =2:length(twin)
    real_label_scaled(i) = median(double(real_label((t < twin(i)) & (t > twin(i-1)))));
end
real_label_scaled(real_label_scaled==0) = REJECT;

%filterSizePred=2;
predict_label_temp=[predict_label(1:filterSizePred);predict_label;predict_label(end-(filterSizePred+1):end)];
for i =1:length(twin)
    predict_label_r_d(i) = median(predict_label_temp(i:i+2*filterSizePred,1));
end
predict_label_r_d=[REJECT;predict_label_r_d(1:end-1)];

% %% visual laughter
% [timeStamp, ~, finalSmoothed] = CreateLaughterStreamFromFaceProps(file,featurePath,filterSizePred);

%% visualize different combinations

[timeStamp, final] = CreateStreamFromFaceProps(file,featurePath);
final=(final(:,1)==3)&(final(:,6)==3)&((final(:,5)==3)|(final(:,4)==3));
%final=(final(:,1)==3)&(final(:,6)==3);
predict_label_temp=[final(1:filterSizePred);final;final(end-(filterSizePred+1):end)];
finalSmoothed = zeros(size(final));
for i =1:length(final)
    finalSmoothed(i) = median(predict_label_temp(i:i+2*filterSizePred,1));
end

% figure;
% subplot(2,1,1);
% bar(timeStamp,finalSmoothed,'b','EdgeColor','None');
% xlim([0 (floor(timeStamp(end)/10)+1)*10]);
% title('happy and mouth open');
% %line([step,step],[0,1],'LineWidth',2,'Color','g');
% 
% subplot(2,1,2);
% bar(twin,real_label_scaled==LAUGHTER,'b','EdgeColor','None');
% hold on
% bar(twin,real_label_scaled==BREATHING,'r','EdgeColor','None');
% xlim([0 (floor(timeStamp(end)/10)+1)*10]);
% title('Real');
% %line([step,step],[0,1],'LineWidth',2,'Color','g');
% hold off;
% drawnow;

%% Plots
figure;
subplot(3,1,1);
bar(twin,predict_label_r_d==LAUGHTER,'b','EdgeColor','None');
hold on;
bar(twin,predict_label_r_d==BREATHING,'r','EdgeColor','None');
xlim([0 (floor(timeStamp(end)/10)+1)*10]);

title('Audio Predicted');
%line([step,step],[0,1],'LineWidth',2,'Color','g');
hold off;

subplot(3,1,2);
bar(twin,real_label_scaled==LAUGHTER,'b','EdgeColor','None');
hold on
bar(twin,real_label_scaled==BREATHING,'r','EdgeColor','None');
xlim([0 (floor(timeStamp(end)/10)+1)*10]);
title('Real');
%line([step,step],[0,1],'LineWidth',2,'Color','g');
hold off;

subplot(3,1,3);
bar(timeStamp,finalSmoothed,'b','EdgeColor','None');
xlim([0 (floor(timeStamp(end)/10)+1)*10]);
title('Visual predicted');
%line([step,step],[0,1],'LineWidth',2,'Color','g');
drawnow;

%% video

videoFile=[audioPath(1:end-31) 'video' audioPath(end-25:end-4) '.mp4'];

%videoFReader = vision.VideoFileReader('./Dataset/Ses04F_impro03.avi', 'AudioOutputPort', 1);
readObj = VideoReader(videoFile);
get(readObj);
%videoFWriter = vision.VideoFileWriter('./Dataset/Ses04F_impro03test.avi','AudioInputPort', 1,'FrameRate',videoFReader.info.VideoFrameRate);
writeObj = VideoWriter(['EXP\' file '.avi'],'Motion JPEG AVI');
writeObj.FrameRate = readObj.FrameRate;
%set(writeObj,'FrameRate',readObj.FrameRate);
open(writeObj);

figure;
set(gcf,'Position',[50 50 1200 600]);
set(gcf,'Renderer','zbuffer');
EOF=0;
x=0;
%while ~isDone(videoFReader)
for k=1:readObj.NumberOfFrames
    %[I, AUDIO] = step(videoFReader);
    I = read(readObj, k);
    subplot(6,1,[1,2,3]);
    imshow(I);
    
    x=x+1/readObj.FrameRate;
    step=min(twin(x<twin));
    
    subplot(6,1,4);
    bar(twin,predict_label_r_d==LAUGHTER,'b','EdgeColor','None');
    hold on;
    bar(twin,predict_label_r_d==BREATHING,'r','EdgeColor','None');
    xlim([0 (floor(timeStamp(end)/10)+1)*10]);
    title('Audio Predicted');hold on;
    if(~isempty(step))
    line([step,step],[0,1],'LineWidth',2,'Color','green');
    end
    hold off;
    subplot(6,1,5);
    bar(twin,real_label_scaled==LAUGHTER,'b','EdgeColor','None');
    hold on
    bar(twin,real_label_scaled==BREATHING,'r','EdgeColor','None');
    xlim([0 (floor(timeStamp(end)/10)+1)*10]);
    title('Real');
    if(~isempty(step))
    line([step,step],[0,1],'LineWidth',2,'Color','green');
    end
    hold off;
    
    subplot(6,1,6);
    bar(timeStamp,finalSmoothed,'b','EdgeColor','None');
    xlim([0 (floor(timeStamp(end)/10)+1)*10]);
    title('Visual predicted');
    if(~isempty(step))
    line([step,step],[0,1],'LineWidth',2,'Color','green');
    end
    hold off;
    
    drawnow;
    M=getframe(gcf);
    %step(videoFWriter,I,AUDIO);
    writeVideo(writeObj,M);
end
%release(videoFWriter);
close(writeObj);
close(readObj);
%release(videoFReader);
end