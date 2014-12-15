clc
close all
clear all

fs = 16000;
Vfs = 120;
K=12;
load PCA_ses1234.mat
Aff=load('AffectBurstsSession1234Cleaned');

winms=750; %in ms
shiftms=250; %frame periodicity in ms

winSize  = winms/1000*fs;
winShift = shiftms/1000*fs;

winSize3d  = winms/1000*Vfs;
winShift3d = shiftms/1000*Vfs;

nfoldCV = 3;
nfold = 10;

% detection 1, recognition 2
classifierType=2;

if(classifierType==1)
    LAUGHTER = 1;
    BREATHING = 1;
    OTHER = 2;
    REJECT = 2;
    axlabels={'Affect Burst','Reject'};
    saveName1='Detection';
else
    LAUGHTER = 1;
    BREATHING = 2;
    OTHER = 3;
    REJECT = 3;
    axlabels={'Laughter','Breathing','Reject'};
    saveName1='Recognition';
end

cRange=[2 4 50];
gRange=[-13 1 -10];
saveName2='Fused';

files=[];
for i=1:4
    file=dir(['../Session' num2str(i) '/dialog/wav']);
    file=file(3:end,:);
    files=[files;file];
end
len=length(files);
% rand_ind = randperm(len);
% save('rand_ind_forclip_ses3half', 'rand_ind');
% load rand_ind_forclip
%% nfold
predictLabels=[]; realLabels=[];
load ./Dataset/clipStats
count=0; clipResults(len).predict_label=[];
for k=1:nfold % Cross training : folding
    test_ind=rand_ind([floor((k-1)*len/nfold)+1:floor(k*len/nfold)]');
    testFiles=files(test_ind);


    load ./Dataset/AffectDataSync
    % Removing Other class
    AffectDataSync(strcmp(extractfield(AffectDataSync,'label'),'Other'))=[];
      for i=1:length(testFiles)
          AffectDataSync(strcmp(extractfield(AffectDataSync,'fileName'),testFiles(i).name(1:end-4)))=[];
      end


    %% label and feature extraction
    LABEL=extractfield(AffectDataSync,'label')';
    label = zeros(length(LABEL),1);
    label(strcmp(LABEL,'Laughter')) = LAUGHTER;
    label(strcmp(LABEL,'Breathing')) = BREATHING;
    label(strcmp(LABEL,'REJECT')) = REJECT;
    %label(strcmp(LABEL,'Other')) = OTHER

    labelList = unique(label);
    NClass = length(labelList);
    datatemp=[]; data=[];
    for i=1:length(AffectDataSync)
        datatemp(i,:)=extract_stats(AffectDataSync(i).data);
        data(i,:)=[datatemp(i,:) extract_stats(AffectDataSync(i).data3d)];
    end

    trainLabelDetect=label;
    trainLabelDetect(trainLabelDetect == LAUGHTER) = 1;
    trainLabelDetect(trainLabelDetect == BREATHING) = 1;
    trainLabelDetect(trainLabelDetect == REJECT) = 2;
    trainDataDetect=data;
    
    trainLabelRec = label;
    trainDataRec=data;
    trainDataRec(trainLabelRec == REJECT,:) = [];
    trainLabelRec(trainLabelRec == REJECT) = [];
  
    [modelDetect, bestParamDetect, gridDetect ]= learn_on_trainingData(trainDataDetect, trainLabelDetect, cRange, gRange, nfoldCV, 0);
    [modelRec, bestParamRec, gridRec ]= learn_on_trainingData(trainDataRec, trainLabelRec, cRange, [-15 1 -12], nfoldCV, 0);

    %% test

    for j=1:length(testFiles)
        count=count+1;
        fileName = testFiles(j).name(1:end-4);
        [y,fs] = wavread(['..\Session',fileName(5),'\dialog\wav\',fileName,'.wav']);
        unseenStats = clipStats(strcmp(extractfield(clipStats,'fileName'),fileName));
        winCount=size(unseenStats.audio,1);
        numberOfFrames = length(y(1:winSize+ winShift*(winCount-1)))*1000/fs;
  
        % feature fusion
        unseenStatsDetect=[unseenStats.audio unseenStats.visual];
        
        [predict_labelDetect, ~ ,prob_valuesDetect] = svmpredict(zeros(size(unseenStatsDetect,1),1), unseenStatsDetect, modelDetect);


        %% ground truth compare
        labelmap = containers.Map;
        labelmap('Laughter') = LAUGHTER;
        labelmap('Breathing') = BREATHING;
        labelmap('Other') = OTHER;
        labelmap('REJECT') = REJECT;

        %AffAnno=Aff.AffectBursts(strcmp(extractfield(Aff.AffectBursts,'fileName'),fileName));

        real_label = GenerateAffectBurstLabelsForSingleFile(Aff.AffectBursts,fileName,numberOfFrames,labelmap);
        
        t=0:1/1000:length(real_label)/1000-1/1000;
        twin=0:shiftms/1000:length(real_label)/1000-shiftms/1000;
        real_label_scaled = zeros(size(predict_labelDetect));
        predict_labelDetect_r_d = zeros(size(predict_labelDetect));

        twin=twin(2:end-1);
        for i =2:length(twin)
            real_label_scaled(i) = median(double(real_label((t < twin(i)) & (t > twin(i-1)))));
        end
        real_label_scaled(real_label_scaled==0) = REJECT;
        
        medianHalf=3;
        predict_label_temp=[predict_labelDetect(1:medianHalf);predict_labelDetect;predict_labelDetect(end-medianHalf+1:end)];
        for i =1:length(twin)
            predict_labelDetect_r_d(i) = median(predict_label_temp(i:i+2*medianHalf,1));
        end
        
        numDilate=2;
        [ predict_labelDetect_r_d ] = Signal_dilate( predict_labelDetect_r_d', numDilate );
        predict_labelDetect_r_d =predict_labelDetect_r_d';
        
        unseenStatsRec = unseenStatsDetect;
        unseenStatsRec = unseenStatsRec(predict_labelDetect_r_d==1,:);
        
        [predict_labelRec, ~ ,prob_valuesRec] = svmpredict(zeros(size(unseenStatsRec,1),1), unseenStatsRec, modelRec);

        predict_label=predict_labelDetect_r_d;
        predict_label(predict_labelDetect_r_d==2)=REJECT;
        predict_label(predict_labelDetect_r_d==1)=predict_labelRec;
        
        
        %% Plots
        figure(1);
        subplot(2,1,1);
        bar(twin,predict_label==LAUGHTER,'b','EdgeColor','None');
        hold on;
        bar(twin,predict_label==BREATHING,'r','EdgeColor','None');

        title('Predicted');
        %line([step,step],[0,1],'LineWidth',2,'Color','g');
        hold off;

        subplot(2,1,2);
        bar(twin,real_label_scaled==LAUGHTER,'b','EdgeColor','None');
        hold on
        bar(twin,real_label_scaled==BREATHING,'r','EdgeColor','None');
        title('Real');
        %line([step,step],[0,1],'LineWidth',2,'Color','g');
        hold off;
        drawnow;

        predictLabels=[predictLabels;predict_label];
        realLabels=[realLabels;real_label_scaled];
        
        clipResults(count).fileName=fileName;
        clipResults(count).predict_label=predict_label;
        clipResults(count).real_label=real_label_scaled;
        clipResults(count).twin=twin;
        clipResults(count).predict_labelDetect=predict_labelDetect_r_d;
        clipResults(count).audio=unseenStats.audio;
        clipResults(count).visual=unseenStats.visual;
       %% clip grab and save
       if(strcmp(fileName,'Ses01F_script02_2') | strcmp(fileName,'Ses04F_impro03'))
           saveas(gcf, ['./EXPproper/ClipOutTest_' fileName], 'fig');
           save(['./EXPproper/ClipOutTest_' fileName],'predict_label','real_label_scaled','twin','fileName');
       end
           
    end
    disp(['done fold ', num2str(k)]);
end

for i =1:NClass
    for j = 1:NClass
    ConfusionMatrix(i,j) = sum(predictLabels(realLabels==i)==j);
    end
end
ConfusionMatrixSensitivity = ConfusionMatrix./(sum(ConfusionMatrix,2)*ones(1,NClass));
ConfusionMatrixPrecision = ConfusionMatrix./(ones(NClass,1)*sum(ConfusionMatrix,1));
Precision = mean(diag(ConfusionMatrixPrecision));
Sensitivity = mean(diag(ConfusionMatrixSensitivity));
ave_acc=sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix));


disp('Frame Wise calculations');
disp(['Acc = ', num2str(sum(realLabels == (predictLabels))/length(predictLabels))]);
disp(['FP= ', num2str(sum(realLabels==2 & (predictLabels==1)))]);
disp(['TP= ', num2str(sum(realLabels==1 & (predictLabels==1)))]);
disp(['TN= ', num2str(sum(realLabels==2 & (predictLabels==2)))]);
disp(['FN= ', num2str(sum(realLabels==1 & (predictLabels==2)))]);
Acc = sum(realLabels == (predictLabels))/length(predictLabels);
FP = sum(realLabels==2 & (predictLabels==1));
TP = sum(realLabels==1 & (predictLabels==1));
TN = sum(realLabels==2 & (predictLabels==2));
FN = sum(realLabels==1 & (predictLabels==2));

ave_acc=100*(TP+TN)/(TP+FP+TN+FN);
precision=100*TP/(TP+FP);
recall=100*TP/(TP+FN);


%saveas(gcf, './EXP/DetectionFused_Unseen', 'fig');
%save ./EXP/DetectionFused_Unseen

save(['./EXPproper/10foldClipOutTestHier_med' num2str(2*medianHalf+1) '_dil' num2str(numDilate)]...
     ,'ConfusionMatrix','ConfusionMatrixPrecision','ConfusionMatrixSensitivity','medianHalf','numDilate');