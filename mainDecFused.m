% clc
% close all
% clear all
% 
% nfoldCV=3;
% %enum{
% LAUGHTER = 1;
% BREATHING = 2;
% OTHER = 3;
% REJECT = 3;
% %}
% 
% pause(0.1);
% 
% % fs = 16000;
% % Vfs = 120;
% % K=12;
% 
% % winms=750; %in ms
% % shiftms=250; %frame periodicity in ms
% % 
% % winSize  = winms/1000*fs;
% % winShift = shiftms/1000*fs;
% % 
% % winSize3d  = winms/1000*Vfs;
% % winShift3d = shiftms/1000*Vfs;
% % 
% % 
% % load AffectBurstsSession123Cleaned
% % load antiAffectBursts
% % load ./Dataset/soundseq.mat
% % load ./Dataset/visseq.mat
% % load PCA
% % 
% % Samples = [AffectBursts;antiAffectBursts(1:round(length(antiAffectBursts)/2))'];
% % 
% % % Feature Extraction
% % idcount=1;
% % AffectDataSync = [];
% % for j  = 1:length(Samples)
% %     datamat=zeros(165,size(visseq(j).data{1,3},1));
% %     for k=1:size(visseq(j).data{1,3},1)
% %         datamat(:,k)=str2double(strsplit(visseq(j).data{1,3}{k}))';
% %     end
% %     i =0;
% %     while winSize3d+ winShift3d*i < size(visseq(j).data{1,3},1)
% %         
% %         PCAcoef = ExtractPCA(datamat(:,1+winShift3d*i:winSize3d+winShift3d*i),U,pcaWmean,K);
% %         AffectDataSync(end+1,:).data3d = PCAcoef;%extract_stats(PCAcoef);
% %         
% %         MFCCs = ExtractMFCC(soundseq(j).data(1+winShift*i:winSize+winShift*i),fs);
% %         AffectDataSync(end,:).data = MFCCs;%extract_stats(MFCCs);
% %         
% %         AffectDataSync(end,:).id = idcount;
% %         AffectDataSync(end,:).label = Samples(j).type;
% %         i  =i + 1;
% %         
% %     end
% %     idcount=idcount+1;
% %     disp(['done with the sample ', num2str(j), ' #wins in total: ' num2str(length(AffectDataSync))]);
% % end
% % 
% % 
% % save('./Dataset/AffectDataSync', 'AffectDataSync');
% 
% load ./Dataset/AffectDataSync
% 
% %% CV
% 
% addpath .\libsvm-3.20
% ind = randperm(length(AffectDataSync))';
% AffectDataSync = AffectDataSync(ind,:);
%  
% LABEL=extractfield(AffectDataSync,'label')';
% label = zeros(length(LABEL),1);
% label(strcmp(LABEL,'Laughter')) = LAUGHTER;
% label(strcmp(LABEL,'Breathing')) = BREATHING;
% label(strcmp(LABEL,'Other')) = OTHER;
% label(strcmp(LABEL,'REJECT')) = REJECT;
% 
% %data=zeros(length(AffectData),length(AffectData(1).data));
% 
% for i=1:length(AffectDataSync)
%     data(i,:)=extract_stats(AffectDataSync(i).data);
% end
% 
% for i=1:length(AffectDataSync)
%     data3d(i,:)=extract_stats(AffectDataSync(i).data3d);
% end
% 
% labelList = unique(label);
% NClass = length(labelList);
%  
% % % #######################
% % % Parameter selection using 3-fold cross validation 3D
% % % #######################
% bestcv3d = 0;
% i =1; j =1;
% for log2c = -2:4:54,
%     for log2g = -10:1:-7,
%         cmd3d = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g), ' -b 1'];
%         cv3d(i,j) = get_cv_ac_bin_probabilistic(label, data3d, cmd3d, nfoldCV);
%         if (cv3d(i,j) >= bestcv3d),
%             bestcv3d = cv3d(i,j); bestc3d = 2^log2c; bestg3d = 2^log2g;
%         end
%         fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv3d(i,j), bestc3d, bestg3d, bestcv3d);
%         j = j+1;
%     end
%     j =1;
%     i = i + 1;
% end
% bestParam3d = ['-q -c ', num2str(bestc3d), ' -g ', num2str(bestg3d), ' -b 1'];
% figure;
% imagesc(cv3d);title('3d CV');
% 
% 
% % % #######################
% % % Parameter selection using 3-fold cross validation Sound
% % % #######################
% bestcv = 0;
% i =1; j =1;
% for log2c = -2:4:54,
%     for log2g = -12:1:-9,
%         cmd = ['-q -c ', num2str(2^log2c), ' -g ', num2str(2^log2g), ' -b 1'];
%         cv(i,j) = get_cv_ac_bin_probabilistic(label, data, cmd, nfoldCV);
%         if (cv(i,j) >= bestcv),
%             bestcv = cv(i,j); bestc = 2^log2c; bestg = 2^log2g;
%         end
%         fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv(i,j), bestc, bestg, bestcv);
%         j = j+1;
%     end
%     j =1;
%     i = i + 1;
% end
% bestParam = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg), ' -b 1'];
% figure;
% imagesc(cv);title('Sound CV');


%% Leave one out test
alfa=0.5;
fusedLabel=[];
prob_template=[1,3,2];
for i=1:max(extractfield(AffectDataSync,'id'))
    testData=data(extractfield(AffectDataSync,'id')==i,:);
    testData3d=data3d(extractfield(AffectDataSync,'id')==i,:);
    testLabel=label(extractfield(AffectDataSync,'id')==i);
    
    trainData=data(extractfield(AffectDataSync,'id')~=i,:);
    trainData3d=data3d(extractfield(AffectDataSync,'id')~=i,:);
    trainLabel=label(extractfield(AffectDataSync,'id')~=i);
    
    model = svmtrain(trainLabel, trainData, bestParam);
    [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, model,'-b 1');
    
    acc(i).accuracy=accuracy(1);
    acc(i).testLabel = testLabel;
    acc(i).predict_label = predict_label;
    acc(i).prob_values = prob_values;
    %prob=[prob;prob_values];
    
    model3d = svmtrain(trainLabel, trainData3d, bestParam3d);
    [predict_label3d, accuracy3d, prob_values3d] = svmpredict(testLabel, testData3d, model3d,'-b 1');
    
    acc3d(i).accuracy=accuracy3d(1);
    acc3d(i).testLabel = testLabel;
    acc3d(i).predict_label = predict_label3d;
    acc3d(i).prob_values = prob_values3d;
    %prob3d=[prob3d;prob_values3d];
    
    fused=acc3d(i).prob_values*(1-alfa)+acc(i).prob_values*alfa;
    [val ind]=max(fused,[],2);
    fusedLabel=[fusedLabel prob_template(ind)];
    disp(['done with ', num2str(i)]);
end

%fused  = decisionFuser( prob, prob3d, 0.5);

%ave=mean(acc(~isnan(acc)));
acc = acc(~isnan(extractfield(acc,'accuracy')));
ave = mean(extractfield(acc,'accuracy'));
fprintf('Ave. Accuracy for audio = %g%%\n', ave);

acc3d = acc3d(~isnan(extractfield(acc3d,'accuracy')));
ave3d = mean(extractfield(acc3d,'accuracy'));
fprintf('Ave. Accuracy for 3d = %g%%\n', ave3d);


%predictLabels = extractfield(acc, 'predict_label');
predictLabels=fusedLabel;
testLabels = extractfield(acc, 'testLabel');
for i =1:NClass
    for j = 1:NClass
    ConfusionMatrix(i,j) = sum(predictLabels(testLabels==i)==j);
    end
end
ConfusionMatrixSensitivity = ConfusionMatrix./(sum(ConfusionMatrix,2)*ones(1,NClass));
ConfusionMatrixPrecision = ConfusionMatrix./(ones(NClass,1)*sum(ConfusionMatrix,1));

% 
% 
% %save(['exp_' num2str(winms) '_' num2str(shiftms) '_D'], 'cv', 'acc', 'ave', 'bestParam', 'bestcv', 'nfoldCV' );

aveFused=100*sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix));
fprintf('Ave. Accuracy = %g%%\n', aveFused);

%% Plots!
figure;
set(gcf,'Position',[50 50 1200 600]);

subplot(1,3,1)
bar3(ConfusionMatrix);
title('Confusion Matrix')
xlabel('GT');
ylabel('P')
subplot(1,3,2)
bar3(ConfusionMatrixSensitivity);
title('Confusion Matrix(Sensitivity)')
xlabel('GT');
ylabel('P');
subplot(1,3,3)

bar3(ConfusionMatrixPrecision);
title('Confusion Matrix(Precision)')
xlabel('GT');
ylabel('P');


% saveas(gcf, './EXP/DetectionFused_1', 'fig');
% save ./EXP/DetectionFused_1

% saveas(gcf, './EXP/RecognitionFused_1', 'fig');
% save ./EXP/RecognitionFused_1

saveas(gcf, './EXP/RecognitionDecFused_3class', 'fig');
save ./EXP/RecognitionDecFused_3class