clc
close all
clear all

% AffectDataSync = createAffectDataSync;
% save('./Dataset/AffectDataSync+sesNumber', 'AffectDataSync');

load ./Dataset/AffectDataSync+sesNumber

% Removing Other class
AffectDataSync(strcmp(extractfield(AffectDataSync,'label'),'Other'))=[];

nfold = 10;

LAUGHTER = 1;
BREATHING = 2;
REJECT = 3;

%% label and feature extraction
LABEL=extractfield(AffectDataSync,'label')';
label = zeros(length(LABEL),1);
label(strcmp(LABEL,'Laughter')) = LAUGHTER;
label(strcmp(LABEL,'Breathing')) = BREATHING;
label(strcmp(LABEL,'REJECT')) = REJECT;

labelList = unique(label);
NClass = length(labelList);
NComponents=4;

%% nfold test
CV(nfold).model=[];
IDs=unique(extractfield(AffectDataSync,'id'));
len=length(IDs);
load rand_ind.mat%rand_ind = randperm(len);
rand_id = IDs(rand_ind);
figure;
for i=1:nfold % nfold test
  train_ind=[];test_ind=[];
  test_id=rand_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]');
  train_id = rand_id;
  train_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]) = [];
  
  for k=1:length(train_id)
      train_ind=[train_ind;find(extractfield(AffectDataSync,'id')==train_id(k))'];
  end
  trainData=AffectDataSync(train_ind,:);
  trainLabel=label(train_ind);
  
  for k=1:length(test_id)
      test_ind=[test_ind;find(extractfield(AffectDataSync,'id')==test_id(k))'];
  end
  testData=AffectDataSync(test_ind,:);
  testLabel=label(test_ind);
  
  [ model ] = trainGMM( trainData, trainLabel, NComponents );
  
%   [CV(i).model, CV(i).bestParam, CV(i).grid ]= learn_on_trainingData(trainData, trainLabel, cRange, gRange, nfoldCV, 0);
%   [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, CV(i).model);
%     acc(i).accuracy=accuracy(1);
%     acc(i).testLabel = testLabel;
%     acc(i).predict_label = predict_label; 
%     subplot(ceil(nfold/5),5,i);imagesc(CV(i).grid);drawnow;
for j=1:length(testData) 
        vectorClasses = zeros(1,NClass);
        for class=1:NClass
            [~,Pos] = posterior(model(class).obj,testData(j).data);           
             suma=0;
             for k=1:NComponents
                 suma = suma + Pos(1,k)*model(class).obj.PComponents(k);
             end
            vectorClasses(class)=suma;
        end
end    
    [v ix] = sort(vectorClasses,'descend');
    if ix(1)==realClass
        success= 1;
    else
        success= 0;
    end

    disp(['done fold ', num2str(i)]);
end


%% confusion matrix
predictLabels = extractfield(acc, 'predict_label');
testLabels = extractfield(acc, 'testLabel');
for i =1:NClass
    for j = 1:NClass
    ConfusionMatrix(i,j) = sum(predictLabels(testLabels==i)==j);
    end
end
ConfusionMatrixSensitivity = ConfusionMatrix./(sum(ConfusionMatrix,2)*ones(1,NClass));
ConfusionMatrixPrecision = ConfusionMatrix./(ones(NClass,1)*sum(ConfusionMatrix,1));

%% plot and metrics
figure;
bar3(ConfusionMatrix');
ax = gca;
set(ax,'XTickLabel',axlabels);
set(ax,'YTickLabel',axlabels);
xlabel('GT');
ylabel('P');

Precision = mean(diag(ConfusionMatrixPrecision));
Sensitivity = mean(diag(ConfusionMatrixSensitivity));

ave_acc=sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix));
title(['Confusion Matrix, ' ' Acc: ' num2str(100*ave_acc) '% Precision: ' num2str(100*mean(Precision)) '% Recall: ' num2str(100*mean(Sensitivity)) '%']);

saveName=['./EXPproper/' saveName1,saveName2];
saveas(gcf, saveName, 'fig');
save(saveName);
    