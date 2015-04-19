function [ AffectDataSync ] = createAffectDataSync_onlyAudio_PNCC(Samples,soundseq,fs)
%UNTÝTLED4 Summary of this function goes here
%   Detailed explanation goes here

addpath C:\Users\Berker\Documents\GitHub\SCE_project
addpath C:\Users\Berker\Documents\GitHub\SCE_project\PNCC

winms=750; %in ms
shiftms=250; %frame periodicity in ms

winSize  = winms/1000*fs;
winShift = shiftms/1000*fs;

        if (fs~=16000)
            for j=1:length(soundseq)
                soundseq(j).data = resample(soundseq(j).data,16000,fs);
            end
            winSize  = winms/1000*16000;
            winShift = shiftms/1000*16000;
        end

% Feature Extraction
idcount=1;
AffectDataSync = [];
for j  = 1:length(Samples)
%     datamat=zeros(165,size(visseq(j).data{1,3},1));
%     datamathead=zeros(6,size(visseq(j).head{1,3},1));
%     for k=1:size(visseq(j).data{1,3},1)
%         datamat(:,k)=str2double(strsplit(visseq(j).data{1,3}{k}))';
%         datamathead(:,k)=str2double(strsplit(visseq(j).head{1,3}{k}))';
%     end
    i =0;
    while winSize+ winShift*i < size(soundseq(j).data,2)
        
%         PCAcoef = ExtractPCA(datamat(:,1+winShift3d*i:winSize3d+winShift3d*i),U,pcaWmean,K);
%         % PCAcoefDelta=deltas(PCAcoef',3)';
%         AffectDataSync(end+1,:).data3d = [PCAcoef datamathead(:,1+winShift3d*i:winSize3d+winShift3d*i)'];%extract_stats(PCAcoef);


        MFCCs = ExtractPNCC(soundseq(j).data(1+winShift*i:winSize+winShift*i)');
        AffectDataSync(end+1,:).data = MFCCs;%extract_stats(MFCCs);
        
        AffectDataSync(end,:).id = idcount;
        AffectDataSync(end,:).label = Samples(j).type;
        AffectDataSync(end,:).fileName = Samples(j).fileName;
        AffectDataSync(end,:).sesNumber = str2num(Samples(j).fileName(5));
        AffectDataSync(end,:).gender = Samples(j).fileName(6);
        i  =i + 1;
        
    end
    idcount=idcount+1;
    disp(['done with the sample ', num2str(j), ' #wins in total: ' num2str(length(AffectDataSync))]);
end


end

