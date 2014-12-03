clc
close all
clear all


path = '../Session4/Session4Annotation/';

files = dir([path,'*.txt']);
AffectBursts = [];

for j = 1: length(files)
    A = fileread([path,files(j).name]);
    data = textscan(A, '%s %f %f %*[^\n]');
    type = data{1};
    sT = data{2};
    eT = data{3};
    
    for i=1:length(data{1})
        
        AffectBursts(end+1,1).type=type{i};
        AffectBursts(end).startTime=sT(i);
        AffectBursts(end).endTime=eT(i);
        AffectBursts(end).fileName=files(j).name(1:end-13);
    end
end
save('newAffectBursts','AffectBursts');

