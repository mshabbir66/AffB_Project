clc
clear all
close all

fs = 16000;
% Vfs=120;

winms=750; %in ms
shiftms=250; %frame periodicity in ms

winSize  = winms/1000*fs;
winShift = shiftms/1000*fs;

% winSize3d  = winms/1000*Vfs;
% winShift3d = shiftms/1000*Vfs;

nfoldCV = 3;
nfold = 10;

LAUGHTER = 1;
BREATHING = 2;
OTHER = 3;
REJECT = 3;
NClass = 3;

cRange=[2 4 50];
gRange=[-13 1 -10];

load ./Dataset/clipStatsResults
len=length(clipStatsResults);

for i=1:len
    real_label=clipStatsResults(i).real_label;
    predict_label=clipStatsResults(i).predict_label;
    real_label(real_label==3 & predict_label~=3)=predict_label(real_label==3 & predict_label~=3);
    clipStatsResults(i).new_real_label=real_label;
end


rand_ind=1:len;%load rand_ind_forclip
%% nfold
predictLabels=[]; realLabels=[];

for k=1:nfold % Cross training : folding
    test_ind=rand_ind([floor((k-1)*len/nfold)+1:floor(k*len/nfold)]');
    train_ind = [1:len]';
    train_ind(test_ind) = [];
    
    trainData=[];trainLabel=[];
    for i=1:length(train_ind)
        % affect burst samples
        fusedData=[clipStatsResults(train_ind(i)).audio clipStatsResults(train_ind(i)).visual];
        affLabels=clipStatsResults(train_ind(i)).new_real_label(clipStatsResults(train_ind(i)).new_real_label~=3);
        trainLabel=[trainLabel;affLabels];
        trainData=[trainData;fusedData(clipStatsResults(train_ind(i)).new_real_label~=3,:)];
        % reject samples
        rejectData=fusedData(clipStatsResults(train_ind(i)).new_real_label==3,:);
        randrej=randi(size(rejectData,1),1,round(length(affLabels)/2))';
        trainLabel=[trainLabel;REJECT*ones(size(randrej))];
        trainData=[trainData;rejectData(randrej,:)];
    end

    trainLabelDetect=trainLabel;
    trainLabelDetect(trainLabelDetect == LAUGHTER) = 1;
    trainLabelDetect(trainLabelDetect == BREATHING) = 1;
    trainLabelDetect(trainLabelDetect == REJECT) = 2;
    trainDataDetect=trainData;
    
    trainLabelRec = trainLabel;
    trainDataRec=trainData;
    trainDataRec(trainLabelRec == REJECT,:) = [];
    trainLabelRec(trainLabelRec == REJECT) = [];
  
    %[modelDetect, bestParamDetect, gridDetect ]= learn_on_trainingData(trainDataDetect, trainLabelDetect, [-6 8 58], [-17 2 -7], nfoldCV, 0);
    [modelRec, bestParamRec, gridRec ]= learn_on_trainingData(trainDataRec, trainLabelRec,[-6 8 66 ], [-21 2 -9], nfoldCV, 0);

    %% test
    testData=[]; testLabel=[];
    for i=1:length(test_ind)
        % feature fusion
        fusedData=[clipStatsResults(test_ind(i)).audio clipStatsResults(test_ind(i)).visual];
        testData=[testData;fusedData];
        testLabel=[testLabel;clipStatsResults(test_ind(i)).new_real_label];
    end
         
        [predict_labelDetect, ~ ,prob_valuesDetect] = svmpredict(zeros(size(testData,1),1), testData, modelDetect);

        
        predict_labelDetect_r_d = zeros(size(predict_labelDetect));
        predict_label_temp=[predict_labelDetect(1:5);predict_labelDetect;predict_labelDetect(end-4:end)];
        for i =1:length(predict_labelDetect)
            predict_labelDetect_r_d(i) = median(predict_label_temp(i:i+10,1));
        end
        
        testDataRec = testData(predict_labelDetect_r_d==1,:);
        
        [predict_labelRec, ~ ,prob_valuesRec] = svmpredict(zeros(size(testDataRec,1),1), testDataRec, modelRec);

        predict_label=predict_labelDetect_r_d;
        predict_label(predict_labelDetect_r_d==2)=REJECT;
        predict_label(predict_labelDetect_r_d==1)=predict_labelRec;
        
        
        %% Plots
        twin=1:length(predict_labelDetect);
        figure(k);
        subplot(2,1,1);
        bar(twin,predict_label==LAUGHTER,'b','EdgeColor','None');
        hold on;
        bar(twin,predict_label==BREATHING,'r','EdgeColor','None');

        title('Predicted');
        %line([step,step],[0,1],'LineWidth',2,'Color','g');
        hold off;

        subplot(2,1,2);
        bar(twin,testLabel==LAUGHTER,'b','EdgeColor','None');
        hold on
        bar(twin,testLabel==BREATHING,'r','EdgeColor','None');
        title('Real');
        %line([step,step],[0,1],'LineWidth',2,'Color','g');
        hold off;
        drawnow;
        
        predictLabels=[predictLabels;predict_label];
        realLabels=[realLabels;testLabel];
       
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


