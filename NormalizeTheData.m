%normalized the Data
clear all
clc

load ./Dataset/AffectDataSyncNew.mat
noS = length(AffectDataSync);

soundData = zeros(noS*88,26);
visualData = zeros(noS*88,12);

for i = 1: noS
soundData((i-1)*88+1:i*88,:) = AffectDataSync(i).data;
visualData((i-1)*88+1:i*88,:) = AffectDataSync(i).data3d;
end

soundDataZ = zscore(soundData,1,1);
visualDataZ = zscore(visualData,1,1);


for i = 1: noS
AffectDataSync(i).data = soundDataZ((i-1)*88+1:i*88,:);
AffectDataSync(i).data3d = visualDataZ((i-1)*88+1:i*88,:);
end

save ./Dataset/AffectDataSyncN.mat AffectDataSync
