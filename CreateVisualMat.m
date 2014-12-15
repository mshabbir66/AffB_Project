% clc
% close all
% clear all
% 
% load AffectBurstsSession1234Cleaned
% load antiAffectBursts
Samples = antiAffectBursts'; %[AffectBursts;antiAffectBursts'];
Vfs=120;

visseq(length(Samples)).data = [];
for i = 1:length(Samples)
    fidv = fopen(['../Session',Samples(i).fileName(5),'/dialog/MOCAP_rotated/',Samples(i).fileName,'.txt'],'r');
    startFrame = round(Vfs*(Samples(i).startTime)/1000);
    if startFrame < 1
        startFrame = 1;
    end
    endFrame = round(Vfs*Samples(i).endTime/1000);
    
    [~]=textscan(fidv,'%d %f %s',startFrame,'Delimiter','\n','Headerlines',2);
    visseq(i).data=textscan(fidv,'%d %f %s',endFrame-startFrame,'Delimiter','\n');
    fclose(fidv);
    
%     [y,fs] = wavread(['..\Session',Samples(i).fileName(5),'\dialog\wav\',Samples(i).fileName,'.wav']);
%     startFrame = round(fs*(Samples(i).startTime)/1000);
%     if startFrame < 1
%         startFrame = 1;
%     end
%     endFrame = round(fs*Samples(i).endTime/1000);
%     switch str2num(Samples(i).fileName(5))
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

%save Dataset/visseq.mat visseq