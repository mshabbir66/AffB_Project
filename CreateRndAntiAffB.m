function [ antiAffectBursts ] = CreateRndAntiAffB( count, sessions, affData, winSize )
%CREATERNDWÝN Summary of this function goes here
%   count: number of windows to be produced
%   sessions: session numbers array
%   affData: Affect burst mat
%   winSize: window size of randomly selected segment (in ms)


randInd=randi(length(sessions),1,count);
sessionList=sessions(randInd);

for i=1:length(sessions)
    fileList{1,i}=ls(['..\Session',num2str(sessions(i)),'\dialog\wav\*.wav']);
end

load(affData);
antiAffectBursts(count).type = [];
for i = 1:count
    
    randFile=fileList{1,sessionList(i)}(randi(size(fileList{1,sessionList(i)},1)),:);
    [y,fs] = wavread(['..\Session',num2str(sessionList(i)),'\dialog\wav\',randFile]);
    
    randFile( randFile == ' ')=[];
    numberOfFrames=length(y)*1000/fs;

    labels = GenerateAffectBurstLabelsForSingleFile(AffectBursts,randFile(1:end-4),numberOfFrames);
    
    
    while(1) %overlap condition for randFile
        startFrame=randi(length(y)-winSize*fs/1000,1);
        endFrame=startFrame+winSize*fs/1000;
        if(~labels(round(startFrame*1000/fs)) |  ~labels(round(endFrame*1000/fs)) | ~labels(round(((startFrame+endFrame)/2)*1000/fs)))
            break;
        end
    end
    
    antiAffectBursts(i).type = 'REJECT';
    antiAffectBursts(i).startTime = startFrame/fs*1000;
    antiAffectBursts(i).endTime = endFrame/fs*1000;
    antiAffectBursts(i).fileName = randFile(1:end-4);
    
    disp(['done with ', num2str(i)]);
end

end

