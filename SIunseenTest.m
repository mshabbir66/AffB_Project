clear all
clc
close all

addpath C:\Users\Berker\Documents\GitHub\SCE_project
videoFile='Ses05F_impro01.avi';
videoFile=['D:\JOKER\Databases\IEMOCAP\Session5\dialog\avi\DivX\' videoFile]; 
audioPath='D:\JOKER\Databases\IEMOCAP\Session5\dialog\wav\Ses05F_impro01.wav';
file=audioPath(end-17:end-4);

medianSize=0;

load SILaughterModel
load C:\Users\Berker\Documents\GitHub\SCE_project\EmotionEvents.mat

LAUGHTER = 1;
BREATHING = 3;
REJECT = 2;

[y,fs] = wavread(audioPath);
winms = 750;
shiftms = 250;
winSize  = winms/1000*fs;
winShift = shiftms/1000*fs;

numberOfFrames=length(y)*1000/fs;
unseenMFCC = [];
i=0;
while winSize+ winShift*i < length(y)
        unseenMFCC(end+1,1).data = ExtractMFCC(y(1+winShift*i:winSize+ winShift*i),fs);
        i  =i + 1;      
end

% normalization

[ unseenMFCC ] = AudioSamplesMFCCNormalization( unseenMFCC );

for i=1:size(unseenMFCC,1)
    unseenStats(i,:) = extract_stats(unseenMFCC(i).data);
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
predict_label_temp=[predict_label(1:medianSize);predict_label;predict_label(end-(medianSize+1):end)];
for i =1:length(twin)
    predict_label_r_d(i) = median(predict_label_temp(i:i+2*medianSize,1));
end
predict_label_r_d=[REJECT;predict_label_r_d(1:end-1)];

%[predict_label_r_d,~] = DilationErosionFilter(predict_label_r_d==LAUGHTER, 3,3);
%[predict_label_r_d,~] = ErosionDilationFilter(predict_label_r_d, 3,3);

%% Plots
figure;
subplot(2,1,1);
bar(twin,predict_label_r_d==LAUGHTER,'b','EdgeColor','None');
hold on;
bar(twin,predict_label_r_d==BREATHING,'r','EdgeColor','None');


title('Audio Predicted');
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

%% video

%videoFile=[audioPath(1:end-31) 'video' audioPath(end-25:end-4) '.mp4'];

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
    subplot(5,1,[1,2,3]);
    imshow(I);
    
    x=x+1/readObj.FrameRate;
    step=min(twin(x<twin));
    
    subplot(5,1,4);
    bar(twin,predict_label_r_d==LAUGHTER,'b','EdgeColor','None');
    hold on;
    bar(twin,predict_label_r_d==BREATHING,'r','EdgeColor','None');

    title('Audio Predicted');hold on;
    if(~isempty(step))
    line([step,step],[0,1],'LineWidth',2,'Color','green');
    end
    hold off;
    subplot(5,1,5);
    bar(twin,real_label_scaled==LAUGHTER,'b','EdgeColor','None');
    hold on
    bar(twin,real_label_scaled==BREATHING,'r','EdgeColor','None');

    title('Real');
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