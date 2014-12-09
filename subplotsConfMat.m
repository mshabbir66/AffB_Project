clc
clear all
close all

% matNames={ 'RecognitionAudio.mat' ; ... 
%            'RecognitionVideo.mat' ; ...
%            'RecognitionFused.mat' ; ...
%            'RecognitionDecFused.mat' ; ...
%            'CascadeRecognitionAudio.mat' ; ... 
%            'CascadeRecognitionVideo.mat' ; ...
%            'CascadeRecognitionFused.mat' ; ...
%            'CascadeRecognitionDecisionFusion.mat' };
% figure('Position',[50 50 1000 600]);
% for i=1:8
%     load(['EXPproper/' matNames{i}],'ConfusionMatrix');
%     subplot(2,4,i);
%     plotConfusionMat;title(['Accuracy: ' num2str(100*sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix)),'%.2f') '%']);
%     axis equal;axis tight;
% end

matNames={ 'CascadeRecognitionAudio.mat' ; ... 
           'CascadeRecognitionVideo.mat' ; ...
           'CascadeRecognitionFused.mat' ; ...
           'CascadeRecognitionDecisionFusion.mat' };
       
figure('Position',[50 50 1000 300]);
for i=1:4
    load(['EXPproper/' matNames{i}],'ConfusionMatrix');
    ConfusionMatrixtemp=ConfusionMatrix;
    ConfusionMatrix=[];
    ConfusionMatrix(1,1)=sum(sum(ConfusionMatrixtemp(1:2,1:2)));
    ConfusionMatrix(2,1)=sum(ConfusionMatrixtemp(3,1:2));
    ConfusionMatrix(1,2)=sum(ConfusionMatrixtemp(1:2,3));
    ConfusionMatrix(2,2)=ConfusionMatrixtemp(3,3);
    subplot(1,4,i);
    plotConfusionMat;title(['Accuracy: ' num2str(100*sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix)),'%.2f') '%']);
    axis equal;axis tight;
end

       
