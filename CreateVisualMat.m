clc
close all
clear all

load Dataset/AffectBurstsSession123Cleaned
Vfs=120;

visseq(length(AffectBursts)).data = [];
for i = 1:length(AffectBursts)
    fidv = fopen(['../Session',AffectBursts(i).fileName(5),'/dialog/MOCAP_rotated/',AffectBursts(i).fileName,'.txt'],'r');
    startFrame = round(Vfs*(AffectBursts(i).startTime)/1000);
    if startFrame < 1
        startFrame = 1;
    end
    endFrame = round(Vfs*AffectBursts(i).endTime/1000);
    
    [~]=textscan(fidv,'%d %f %s',startFrame,'Delimiter','\n','Headerlines',2);
    visseq(i).data=textscan(fidv,'%d %f %s',endFrame-startFrame,'Delimiter','\n');
    fclose(fidv);
    
%     [y,fs] = wavread(['..\Session',AffectBursts(i).fileName(5),'\dialog\wav\',AffectBursts(i).fileName,'.wav']);
%     startFrame = round(fs*(AffectBursts(i).startTime)/1000);
%     if startFrame < 1
%         startFrame = 1;
%     end
%     endFrame = round(fs*AffectBursts(i).endTime/1000);
%     switch str2num(AffectBursts(i).fileName(5))
%         case 1
%             y = y(startFrame:endFrame,1);
%         case 2
%             y = y(startFrame:endFrame,2);
%         case 3
%             y = y(startFrame:endFrame,2);
%     end  
%     soundseq{i} = y';
    disp(['done with ', num2str(i)]);
    
end
