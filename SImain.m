%%%%%%%% Speaker independent laughter analysis %%%%%%%%

clear all;clc;close all;
addpath C:\Users\Berker\Documents\GitHub\SCE_project
addpath C:\Users\HP\Documents\GitHub\SCE_project
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


% aveDurationAVlaughterCycle=sum(extractfield(AVlaughterCycleAffectBursts,'endTime')-extractfield(AVlaughterCycleAffectBursts,'startTime'))/length(AVlaughterCycleAffectBursts);
% [ antiAffectBurstsforAVlaughterCycle ] = CreateRndAntiAffB( length(AVlaughterCycleAffectBursts), [1 2 3 4], AffectBursts, aveDurationAVlaughterCycle );
% antiAffsoundseqforAVlaughterCycle=CreateAudioMat(antiAffectBurstsforAVlaughterCycle);
%[ antiAffectDataSyncMFCC_AVLaughterCycle ] = createAffectDataSync_onlyAudio_MFCC(antiAffectBurstsforAVlaughterCycle,antiAffsoundseqforAVlaughterCycle,fs);

load ('Dataset/SILaughterData.mat',...
'randomIndex',...
'AffectDataSyncMFCC',...
'antiAffectDataSyncMFCC',...
'AffectDataSyncMFCC_AVLaughterCycle',...
'antiAffectDataSyncMFCC_AVLaughterCycle');
%'AffectDataSyncPNCC',...
%'antiAffectDataSyncPNCC');

%% MFCC segmented session out

AffectDataSync=[AffectDataSyncMFCC;antiAffectDataSyncMFCC];
AffectDataSync = AffectDataSync(randomIndex,:);

[ConfusionMatrixMFCCsessionOut]=AudioSessionOut(AffectDataSync);

%% PNCC segmented session out

AffectDataSync=[AffectDataSyncPNCC;antiAffectDataSyncPNCC];
AffectDataSync = AffectDataSync(randomIndex,:);

[ConfusionMatrixPNCCsessionOut]=AudioSessionOut(AffectDataSync);

%% MFCC speaker out

AffectDataSync=[AffectDataSyncMFCC;antiAffectDataSyncMFCC];
AffectDataSync = AffectDataSync(randomIndex,:);

[ConfusionMatrixMFCCspeakerOut,acc]=AudioSpeakerOut(AffectDataSync);

%% PNCC speaker out

AffectDataSync=[AffectDataSyncPNCC;antiAffectDataSyncPNCC];
AffectDataSync = AffectDataSync(randomIndex,:);

[ConfusionMatrixPNCCspeakerOut,acc]=AudioSpeakerOut(AffectDataSync);

%% MFCC speaker out with AVLaughterCycle

% normalization (mean and variance)
temp=max(extractfield(AffectDataSyncMFCC,'id'));
for i=1:length(antiAffectDataSyncMFCC)
antiAffectDataSyncMFCC(i).id=antiAffectDataSyncMFCC(i).id+temp;
end
temp=max(extractfield(antiAffectDataSyncMFCC,'id'));
for i=1:length(antiAffectDataSyncMFCC_AVLaughterCycle)
antiAffectDataSyncMFCC_AVLaughterCycle(i).id=antiAffectDataSyncMFCC_AVLaughterCycle(i).id+temp;
end

[ AudioSamplesMFCC_IEMOCAP ] = AudioSamplesMFCCNormalization( [AffectDataSyncMFCC;antiAffectDataSyncMFCC;antiAffectDataSyncMFCC_AVLaughterCycle] );
[ AffectDataSyncMFCC_AVLaughterCycle ] = AudioSamplesMFCCNormalization( AffectDataSyncMFCC_AVLaughterCycle );

temp=max(extractfield(AudioSamplesMFCC_IEMOCAP,'id'));
temp1=max(extractfield(AudioSamplesMFCC_IEMOCAP,'sesNumber'));
for i=1:length(AffectDataSyncMFCC_AVLaughterCycle)
AffectDataSyncMFCC_AVLaughterCycle(i).id=AffectDataSyncMFCC_AVLaughterCycle(i).id+temp;
AffectDataSyncMFCC_AVLaughterCycle(i).sesNumber=AffectDataSyncMFCC_AVLaughterCycle(i).sesNumber+temp1;
end

AffectDataSync=[ AudioSamplesMFCC_IEMOCAP;AffectDataSyncMFCC_AVLaughterCycle ];
clear ('AffectDataSyncMFCC','antiAffectDataSyncMFCC','antiAffectDataSyncMFCC_AVLaughterCycle','AudioSamplesMFCC_IEMOCAP','AffectDataSyncMFCC_AVLaughterCycle');
randomIndexforAVlaughterCycle = randperm(length(AffectDataSync));
AffectDataSync = AffectDataSync(randomIndexforAVlaughterCycle,:);

load Dataset/SILaughterData_addition.mat

%[ConfusionMatrixMFCCspeakerOut_IEMOCAP,acc]=AudioNFoldSpeaker(AffectDataSync,5);

[ConfusionMatrixSessionClipsOut,acc]=SITestSessionClipsOut(AffectDataSync,4);