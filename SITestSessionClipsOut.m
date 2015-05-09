function [ConfusionMatrix,acc]=SITestSessionClipsOut(AffectDataSyncBase,sessions);

fs = 16000;
Aff=load('AffectBurstsSession1234Cleaned');

winms=750; %in ms
shiftms=250; %frame periodicity in ms

winSize  = winms/1000*fs;
winShift = shiftms/1000*fs;

nfoldCV = 3;

LAUGHTER = 1;
REJECT = 2;


cRange=[-2 4 42];
gRange=[-15 1 -9];


files=[];
for i=1:4
    file=dir(['../Session' num2str(i) '/dialog/wav']);
    file=file(3:end,:);
    files=[files;file];
end
len=length(files);
%% 
predictLabels=[]; realLabels=[];
clipStats(len).audio=[];
count=0;
acc(sessions).testLabel=[];
for k=1:sessions % Cross training : folding
    
    testFiles=dir(['../Session' num2str(k) '/dialog/wav']);
    testFiles=file(3:end,:);

    AffectDataSync=AffectDataSyncBase;

    for i=1:length(testFiles)
        AffectDataSync(strcmp(extractfield(AffectDataSync,'fileName'),testFiles(i).name(1:end-4)))=[];
    end
    
    %% label and feature extraction
    LABEL=extractfield(AffectDataSync,'label')';
    label = zeros(length(LABEL),1);
    label(strcmp(LABEL,'Laughter')) = LAUGHTER;
    label(strcmp(LABEL,'REJECT')) = REJECT;

    labelList = unique(label);
    NClass = length(labelList);
    datatemp=[]; data=[];
    parfor i=1:length(AffectDataSync)
        data(i,:)=extract_stats(AffectDataSync(i).data);
        disp(['Extracted stats for sample ', num2str(i)]);
    end

    [model, bestParam, grid ]= learn_on_trainingData(data, label, cRange, gRange, nfoldCV, 0);

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
        while winSize+ winShift*i < length(y) & winSize3d+ winShift3d*i < size(text{1,3},1)
            PCAcoef = ExtractPCA(datamat(:,1+winShift3d*i:winSize3d+winShift3d*i),U,pcaWmean,K);
            MFCCs = ExtractMFCC(y(1+winShift*i:winSize+ winShift*i),fs);
            clipStats(count).audio(end+1,:)=extract_stats(MFCCs); 
            clipStats(count).visual(end+1,:)=extract_stats(PCAcoef);
            i = i + 1;
        end
        if ((winSize3d+ winShift3d*i < size(text{1,3},1))==0)
            numberOfFrames = length(y(1:winSize+ winShift*(i-1)))*1000/fs;
            y =y(1:winSize+ winShift*(i-1));
        end
        clipStats(count).fileName=fileName;
        % feature fusion
        unseenStats=[clipStats(count).audio clipStats(count).visual];
        
        [predict_label, ~ ,prob_values] = svmpredict(zeros(size(unseenStats,1),1), unseenStats, model);

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

        predict_label_temp=[predict_label(1:5);predict_label;predict_label(end-6:end)];
        for i =1:length(twin)
            predict_label_r_d(i) = median(predict_label_temp(i:i+10,1));
        end

        %% Plots
        figure(count);
        subplot(2,1,1);
        bar(twin,predict_label_r_d==LAUGHTER,'b','EdgeColor','None');
        title('Predicted');
        %line([step,step],[0,1],'LineWidth',2,'Color','g');


        subplot(2,1,2);
        bar(twin,real_label_scaled==LAUGHTER,'b','EdgeColor','None');
        title('Real');
        %line([step,step],[0,1],'LineWidth',2,'Color','g');
        drawnow;

        predictLabels=[predictLabels;predict_label_r_d];
        realLabels=[realLabels;real_label_scaled];

    end
    acc(k).testFiles=testFiles;
    acc(k).testLabel = realLabels;
    acc(k).predict_label = predictLabels;
    acc(k).session=k;
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

end


