clc
close all
clear all

load AffectBurstsSession123Cleaned
soundseq{length(AffectBursts)} = [];
for i = 1:length(AffectBursts)
    [y,fs] = wavread(['..\Session',AffectBursts(i).fileName(5),'\dialog\wav\',AffectBursts(i).fileName,'.wav']);
    startFrame = round(fs*(AffectBursts(i).startTime)/1000);
    if startFrame < 1
        startFrame = 1;
    end
    endFrame = round(fs*AffectBursts(i).endTime/1000);
    switch str2num(AffectBursts(i).fileName(5))
        case 1
            y = y(startFrame:endFrame,1);
        case 2
            y = y(startFrame:endFrame,2);
        case 3
            y = y(startFrame:endFrame,2);
    end  
    soundseq(i).data = y';
    disp(['done with ', num2str(i)]);
    
end
