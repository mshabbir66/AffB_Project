clear all
close all
clc

load ('AffectBurstsSession1234Cleaned.mat','AffectBursts');
%AffectBursts = AffectBurstsCleaned;

%type = strcmp(extractfield(AffectBursts,'type'),'Laughter');
ind = 1:length(AffectBursts);%find(type);
%ind = find(type);

session = 'Session1';
audioPath = ['../',session,'/dialog/wav/'];
videoPath = ['../',session,'/dialog/avi/DivX/'];

soundFrameDuration = 10; %in ms
soundFrequency = 16000;
soundFrameSize = soundFrequency/1000*soundFrameDuration;

% sampleLaughter=46 Ses01F_script01_3, sampleBreathing=36 Ses01F_script01_1

i=46;
vec = ones(1,length(AffectBursts));


while (i<=length(ind))
    figure(1);
    set(1,'Position',[50 50 600 500])
%     figure(2);
%     set(2,'position',[692    49   667   636])
    A =input('(1)To go forward, (2)To go back. (3)To remove current, (4)To add current:');
    if ~isempty(A)
        if (A ==1)
            i =i+1;
        elseif(A==2)
            i=i-1;
        elseif(A==3)
            vec(i) = 0;
            continue;
        elseif(A==4)
            vec(i) = 1;
            continue
        end
    end

[y,fs] = wavread([audioPath,AffectBursts(ind(i)).fileName,'.wav']);

startFrame = round(fs/1000*(AffectBursts(ind(i)).startTime));
endFrame = round(fs/1000*AffectBursts(ind(i)).endTime);
y = y(startFrame:endFrame,1);
t = 0:1/fs: (endFrame-startFrame)/fs;

y=y/max(y);
figure(1)
clf
ax = [];
subplot(3,1,2)
plot(t,y);  
axis tight;
ax(end+1)=gca; 

subplot(3,1,3)

spectrogram(y,320,160,256,fs,'yaxis');
xlabel('Time(sec)');
%surf(log10(abs(A)),'EdgeColor','none');
%view([0 90]);
axis tight
ax(end+1)=gca; 


xyloObj = VideoReader([videoPath,AffectBursts(ind(i)).fileName,'.avi']);
nFrames = xyloObj.FrameRate;
vidHeight = xyloObj.Height;
vidWidth = xyloObj.Width;
totalFramesofFile=xyloObj.NumberOfFrames;

startFrame=round(nFrames/1000*(AffectBursts(ind(i)).startTime));
endFrame=round(nFrames/1000*(AffectBursts(ind(i)).endTime));
TotalFrames = endFrame-startFrame;  


for k = 0 : TotalFrames
    tempVid(:,:,:,k+1) = read(xyloObj,startFrame+k);
end

k=0;



subplot(3,1,1);
imshow([tempVid(115:end-112,1:362,:,round(TotalFrames/5)) ...
    tempVid(115:end-112,8:362,:,round(2*TotalFrames/5))...
    tempVid(115:end-112,8:362,:,round(3*TotalFrames/5))...
    tempVid(115:end-112,8:362,:,round(4*TotalFrames/5))]);
disp(['#frame: ' num2str(startFrame+round(TotalFrames/5)),...
    '#frame: ' num2str(startFrame+round(2*TotalFrames/5)),...
    '#frame: ' num2str(startFrame+round(3*TotalFrames/5)),...
    '#frame: ' num2str(startFrame+round(4*TotalFrames/5))]);
%title(['AffectBurst No.',num2str(ind(i)),' type:',AffectBursts(ind(i)).type]);


sound(y,fs);

% for k = 0 : TotalFrames
%     time =tic;
%     subplot(3,1,1)
%     imshow(tempVid(:,:,:,k+1));
%     title(['AffectBurst No.',num2str(ind(i)),' type:',AffectBursts(ind(i)).type]);
%     drawnow;
%     time =  toc(time);
%     pause(1/nFrames-time)
% end

end
