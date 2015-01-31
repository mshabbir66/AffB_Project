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

cRange=[-2 4 46];
gRange=[-13 1 -10];
saveName2='Fused';

files=[];
for i=1:4
    file=dir(['../Session' num2str(i) '/dialog/wav/*.wav']);
    
    files=[files;file];
end
len=length(files);
% rand_ind = randperm(len);
% save('rand_ind_forclip', 'rand_ind');
load rand_ind_forclip_ses2half
%% nfold
predictLabels=[]; realLabels=[];
clipStats(len).AffectDataSync=[];
count=0;
for k=1:10 % Cross training : folding
    test_ind=rand_ind([floor((k-1)*len/nfold)+1:floor(k*len/nfold)]');
    testFiles=files(test_ind);


    load ./Dataset/AffectDataSyncN
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
    %datatemp=[]; data=[];
    %for i=1:length(AffectDataSync)
    %    datatemp(i,:)=extract_stats(AffectDataSync(i).data);
    %    data(i,:)=[datatemp(i,:) extract_stats(AffectDataSync(i).data3d)];
    %end

    %[model, bestParam, grid ]= learn_on_trainingData(data, label, cRange, gRange, nfoldCV, 0);

    trainData=AffectDataSync;
    trainLabel=label;
    
    trainDataCell = struct2cell(trainData);
    %select sound
    trainDataSound = trainDataCell(2,:)';
    trainData3D = trainDataCell(1,:)';
    
    for kr = 1:length(trainData)
        trainDataSound{kr} = trainDataSound{kr}';
        trainData3D{kr} = trainData3D{kr}';
    end
    
    [CV.model ]= trainHMMGMM(trainDataSound, trainLabel,4,4);
    [CV3D.model ]= trainHMMGMM(trainData3D, trainLabel,4,4);
   
    
    %% test

    for j=1:length(testFiles)
        fileName = testFiles(j).name(1:end-4);
        [y,fs] = wavread(['..\Session',fileName(5),'\dialog\wav\',fileName,'.wav']);

        switch str2num(fileName(5))
            case 1
                y = y(:,1);
            case 2
                y = y(:,2);
            case 3
                y = y(:,2);
            case 4
                y = y(:,2);
        end

        fidv = fopen(['../Session',fileName(5),'/dialog/MOCAP_rotated/',fileName ,'.txt'],'r');
        text=textscan(fidv,'%d %f %s','Delimiter','\n','Headerlines',2);
        fclose(fidv);

        datamat=zeros(165,size(text{1,3},1));
        for i=1:size(text{1,3},1)
            datamat(:,i)=str2double(strsplit(text{1,3}{i}))';
        end

        numberOfFrames=length(y)*1000/fs;
        unseenStats = [];
        i=0;
        count=count+1;
        while winSize+ winShift*i < length(y) && winSize3d+ winShift3d*i < size(text{1,3},1)
            PCAcoef = ExtractPCA(datamat(:,1+winShift3d*i:winSize3d+winShift3d*i),U,pcaWmean,K);
            MFCCs = ExtractMFCC(y(1+winShift*i:winSize+ winShift*i),fs);
            clipStats(count).AffectDataSync(end+1,1).Audio=MFCCs; 
            clipStats(count).AffectDataSync(end,1).Video=PCAcoef;
            i = i + 1;
        end
        if ((winSize3d+ winShift3d*i < size(text{1,3},1))==0)
            numberOfFrames = length(y(1:winSize+ winShift*(i-1)))*1000/fs;
            y =y(1:winSize+ winShift*(i-1));
        end
        clipStats(count).fileName=fileName;
        
        % feature fusion
        %% check the order here!!
        testData=clipStats(count).AffectDataSync;
        testDataCell = struct2cell(testData);
        %select sound
        testDataSound = testDataCell(1,:)';
        testData3D = testDataCell(2,:)';
         
        for kr = 1:length(testData)
            testDataSound{kr} = testDataSound{kr}';
            testData3D{kr} = testData3D{kr}';
        end
       
        [~,~, prob_valuesSound] = testHMMGMM(0, testDataSound, CV.model);
        [~,~, prob_values3D] = testHMMGMM(0, testData3D, CV3D.model);
        predict_label = decisionFuserModified( prob_valuesSound, prob_values3D, 0.5);
        [~, predict_label]=max(predict_label,[],2);
        
        %unseenStats=[clipStats(count).audio clipStats(count).visual];
        
        %[predict_label, ~ ,prob_values] = svmpredict(zeros(size(unseenStats,1),1), unseenStats, model);

        %% ground truth compare
        labelmap = containers.Map;
        labelmap('Laughter') = LAUGHTER;
        labelmap('Breathing') = BREATHING;
        labelmap('Other') = OTHER;
        labelmap('REJECT') = REJECT;

        AffAnno=Aff.AffectBursts(strcmp(extractfield(Aff.AffectBursts,'fileName'),fileName));

        real_label = GenerateAffectBurstLabelsForSingleFile(AffAnno,fileName,numberOfFrames,labelmap);
        
        t=0:1/1000:length(real_label)/1000-1/1000;
        twin=0:shiftms/1000:length(real_label)/1000-shiftms/1000;
        real_label_scaled = zeros(size(predict_label));
        predict_label_r_d = zeros(size(predict_label));

        twin=twin(2:end-1);
        for i =2:length(twin)
            real_label_scaled(i) = median(double(real_label((t < twin(i)) & (t > twin(i-1)))));
        end
        real_label_scaled(real_label_scaled==0) = REJECT;
        
%         predict_comb=(predict_label~=REJECT);
%         mask=zeros(size(predict_label));
%         for i =6:length(twin)-5
%             mask(i) = median(predict_comb(i-5:i+5,1));
%         end
%         predict_label_r_d=predict_label & mask;

%         for i =6:length(twin)-5
%             predict_label_r_d(i) = median(predict_label(i-5:i+5,1));
%         end
% 
%         predict_label_temp=[predict_label(1:5);predict_label;predict_label(end-6:end)];
%         for i =1:length(twin)
%             predict_label_r_d(i) = median(predict_label_temp(i:i+10,1));
%         end

        %% Plots
        figure(count);
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

