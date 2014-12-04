function [ AffectDataSync ] = createAffectDataSync
%UNTÝTLED4 Summary of this function goes here
%   Detailed explanation goes here
fs = 16000;
Vfs = 120;
K=12;

winms=750; %in ms
shiftms=250; %frame periodicity in ms

winSize  = winms/1000*fs;
winShift = shiftms/1000*fs;

winSize3d  = winms/1000*Vfs;
winShift3d = shiftms/1000*Vfs;


load AffectBurstsSession1234Cleaned
load antiAffectBursts
load ./Dataset/soundseq.mat
load ./Dataset/visseq.mat
load PCA_ses1234.mat

Samples = [AffectBursts;antiAffectBursts(1:round(length(antiAffectBursts)/2))'];

% Feature Extraction
idcount=1;
AffectDataSync = [];
for j  = 1:length(Samples)
    datamat=zeros(165,size(visseq(j).data{1,3},1));
    for k=1:size(visseq(j).data{1,3},1)
        datamat(:,k)=str2double(strsplit(visseq(j).data{1,3}{k}))';
    end
    i =0;
    while winSize3d+ winShift3d*i < size(visseq(j).data{1,3},1)
        
        PCAcoef = ExtractPCA(datamat(:,1+winShift3d*i:winSize3d+winShift3d*i),U,pcaWmean,K);
        AffectDataSync(end+1,:).data3d = PCAcoef;%extract_stats(PCAcoef);
        
        MFCCs = ExtractMFCC(soundseq(j).data(1+winShift*i:winSize+winShift*i),fs);
        AffectDataSync(end,:).data = MFCCs;%extract_stats(MFCCs);
        
        AffectDataSync(end,:).id = idcount;
        AffectDataSync(end,:).label = Samples(j).type;
        AffectDataSync(end,:).sesNumber = str2num(Samples(j).fileName(5));
        i  =i + 1;
        
    end
    idcount=idcount+1;
    disp(['done with the sample ', num2str(j), ' #wins in total: ' num2str(length(AffectDataSync))]);
end


end

