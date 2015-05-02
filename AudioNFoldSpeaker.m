function [ConfusionMatrix,acc]=AudioNFoldSpeaker(AffectDataSync,nfold)

nfoldCV=3;

%enum{
LAUGHTER = 1;
REJECT = 2;
%}

%% CV
 
LABEL=extractfield(AffectDataSync,'label')';
label = zeros(length(LABEL),1);
label(strcmp(LABEL,'Laughter')) = LAUGHTER;
label(strcmp(LABEL,'REJECT')) = REJECT;


parfor i=1:length(AffectDataSync)
    data(i,:)=extract_stats(AffectDataSync(i).data); %[datatemp(i,:) extract_stats(AffectDataSync(i).data3d)];
    disp(['Extracted stats for sample ', num2str(i)]);
end


labelList = unique(label);
NClass = length(labelList);

cRange=[-2 4 42];
gRange=[-15 1 -9];

% %% Leave one Session out test
% 
temp=extractfield(AffectDataSync,'sesNumber');
gender=extractfield(AffectDataSync,'gender');
for i=1:length(temp)
sesNum(i)=temp{i};
end

cnt=0;
genCode=['M','F'];
sessionList=unique(sesNum);
rand_ind = randperm(length(sessionList));
sessionList=sessionList(rand_ind);
for k=sessionList
    for l=1:2
        mask =(sesNum==k)&strcmp(gender,genCode(l));
        if(sum(mask)==0)
            continue;
        end
        cnt=cnt+1;
        for i=1:length(mask)
            if(mask(i))
            AffectDataSync(i).speaker=cnt;
            end
        end
    end
end

speakers=unique(extractfield(AffectDataSync,'speaker'));
len=length(speakers);
load speakerrand %rand_ind = randperm(len);
rand_id=speakers(rand_ind);
for i=1:nfold % nfold test
  train_ind=[];test_ind=[];
  test_id=rand_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]');
  train_id = rand_id;
  train_id([floor((i-1)*len/nfold)+1:floor(i*len/nfold)]) = [];
        
  for k=1:length(train_id)
      train_ind=[train_ind;find(extractfield(AffectDataSync,'speaker')==train_id(k))'];
  end
  trainData=data(train_ind,:);
  trainLabel=label(train_ind);
  
  for k=1:length(test_id)
      test_ind=[test_ind;find(extractfield(AffectDataSync,'speaker')==test_id(k))'];
  end
  testData=data(test_ind,:);
  testLabel=label(test_ind);
        

        [ model, bestParam, cv ] = learn_on_trainingData(trainData, trainLabel, cRange, gRange, nfoldCV, 0 );


        [predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, model);

        acc(i).accuracy=accuracy(1);
        acc(i).testLabel = testLabel;
        acc(i).predict_label = predict_label;
        acc(i).speakers=test_id;
        disp(['done with ', num2str(i)]);
   
end


acc = acc(~isnan(extractfield(acc,'accuracy')));
%ave = mean(extractfield(acc,'accuracy'));
%fprintf('Ave. Accuracy = %g%%\n', ave);


predictLabels = extractfield(acc, 'predict_label');
testLabels = extractfield(acc, 'testLabel');
for i =1:NClass
    for j = 1:NClass
    ConfusionMatrix(i,j) = sum(predictLabels(testLabels==i)==j);
    end
end

%% Plots and metrics!
TP=ConfusionMatrix(1,1);
FP=ConfusionMatrix(2,1);
TN=ConfusionMatrix(2,2);
FN=ConfusionMatrix(1,2);

ave_acc=100*(TP+TN)/(TP+FP+TN+FN);
precision=100*TP/(TP+FP);
recall=100*TP/(TP+FN);
%ave=100*sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix));
fprintf('Ave. Accuracy = %g%%\n', ave_acc);

figure;
set(gcf,'Position',[50 50 1200 600]);

%subplot(1,3,1)
bar3(ConfusionMatrix');
title(['Confusion Matrix, ' ' Acc: ' num2str(ave_acc) '% Precision: ' num2str(precision) '% Recall: ' num2str(recall) '%']);
ax=gca;
set(ax,'XTickLabel',{'Affect Burst','Reject'});
set(ax,'YTickLabel',{'Affect Burst','Reject'});
xlabel('GT');
ylabel('P')

end

