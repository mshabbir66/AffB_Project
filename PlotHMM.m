%%
close all
clear all


matFiles = dir('./EXPproper/*.mat');
matFiles2= dir('D:\Joker\AffB_Project\EXPproper\berker\*.mat');

[X,Y] = meshgrid(2.^(0:3),2.^(1:5));
Z = zeros(size(X));

for i = 1:length(matFiles)
load (['./EXPproper/',matFiles(i).name],'ConfusionMatrix','ave_acc','noS','noM')
indS = log(noS)/log(2)+1;
indM = log(noM)/log(2);
Z(indM,indS) = ave_acc;
end

for i = 1:length(matFiles2)
load (['D:\Joker\AffB_Project\EXPproper\berker\',matFiles2(i).name],'com','ConfusionMatrix','maxind','ave_acc')

indM = log(com)/log(2);

Z(indM,1) = ave_acc(maxind);

end
