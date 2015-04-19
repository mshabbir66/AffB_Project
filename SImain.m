%%%%%%%% Speaker independent laughter analysis %%%%%%%%

clear all;clc;close all;
addpath C:\Users\Berker\Documents\GitHub\SCE_project

% %% preparing IEMOCAP data
% load('AffectBurstsSession1234Cleaned.mat','AffectBursts');
% AffectBursts=AffectBursts(strcmp(extractfield(AffectBursts,'type'),'Laughter'));
% 
% Affsoundseq=CreateAudioMat(AffectBursts);
% %visseq=CreateVisualMat(AffectBursts);
% 
% aveDuration=sum(extractfield(AffectBursts,'endTime')-extractfield(AffectBursts,'startTime'))/length(AffectBursts);
% [ antiAffectBursts ] = CreateRndAntiAffB( length(AffectBursts), [1 2 3 4], AffectBursts, aveDuration );
% antiAffsoundseq=CreateAudioMat(antiAffectBursts);
% 
% [ AffectDataSyncPNCC ] = createAffectDataSync_onlyAudio_PNCC(AffectBursts,Affsoundseq,fs);
% [ antiAffectDataSyncPNCC ] = createAffectDataSync_onlyAudio_PNCC(antiAffectBursts,antiAffsoundseq,fs);
% 
% [ AffectDataSyncMFCC ] = createAffectDataSync_onlyAudio_MFCC(AffectBursts,Affsoundseq,fs);
% [ antiAffectDataSyncMFCC ] = createAffectDataSync_onlyAudio_MFCC(antiAffectBursts,antiAffsoundseq,fs);

%%
load Dataset/SILaughterData.mat

AffectDataSync=[AffectDataSyncMFCC;antiAffectDataSyncMFCC];
AffectDataSync = AffectDataSync(randomIndex,:);

[ConfusionMatrix]=AudioSessionOut(AffectDataSync);