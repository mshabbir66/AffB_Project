%%
close all
clear all


matFiles = dir('./EXPproper/*.mat');
matFiles2= dir('D:\Joker\AffB_Project\EXPproper\berker\*.mat');

[X,Y] = meshgrid(2.^(0:3),2.^(1:5));
Z = zeros(size(X));

for i = 1:length(matFiles)
    if ~strcmp(matFiles(i).name,'TestFoldGMM.mat')
        load (['./EXPproper/',matFiles(i).name],'ConfusionMatrix','ave_acc','noS','noM')
        indS = log(noS)/log(2)+1;
        indM = log(noM)/log(2);
        Z(indM,indS) = ave_acc;
    end
end

for i = 1:length(matFiles2)
    load (['D:\Joker\AffB_Project\EXPproper\berker\',matFiles2(i).name],'com','ConfusionMatrix','maxind','ave_acc')
    
    indM = log(com)/log(2);
    
    Z(indM,1) = ave_acc(maxind);
    
end


mat = Z;           %# A 5-by-5 matrix of random values from 0 to 1

imagesc(mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(mat(:),'%0.3f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding


[x,y] = meshgrid(1:4,1:5);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

 set(gca,'XTick',1:5,...                         %# Change the axes tick marks
         'XTickLabel',{'1','2','4','8','E'},...  %#   and tick labels
         'YTick',1:5,...
         'YTickLabel',{'2','4','8','16','32'},...%    
         'TickLength',[0 0]);
     
     xlabel('Number of States')
     ylabel ('Number of Gaussian Mixtures')
     title('Average accuracy on 10-fold presegmented affect bursts (HMM)')