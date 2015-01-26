function [ model, bestCom, cv ] = CVtrainHMM(trainData,trainDataCell, trainLabel, comRange, nfoldCV)

cv(nfoldCV).acc=[];
NClass=length(unique(trainLabel));
bestcv = 0;
i =1;
for N = comRange
    %         cv(i) = get_cv_ac_bin(trainLabel, trainData, cmd, nfoldCV);
    
    com=2^N;
    ac = 0;
    IDs=unique(extractfield(trainData,'id'));
    len=length(IDs);
    rand_ind = randperm(len);
    rand_id = IDs(rand_ind);
    for j=1:nfoldCV % Cross training : folding
        train_ind=[];test_ind=[];
        test_id=rand_id([floor((j-1)*len/nfoldCV)+1:floor(j*len/nfoldCV)]');
        train_id = rand_id;
        train_id([floor((j-1)*len/nfoldCV)+1:floor(j*len/nfoldCV)]) = [];
        
        for k=1:length(train_id)
            train_ind=[train_ind;find(extractfield(trainData,'id')==train_id(k))'];
        end
        
        for k=1:length(test_id)
            test_ind=[test_ind;find(extractfield(trainData,'id')==test_id(k))'];
        end
        
      
        [ model ] = trainHMMGMM(trainDataCell(train_ind), trainLabel(train_ind), com);
        [predict_label,~, ~] = testHMMGMM(trainLabel(test_ind), trainDataCell(test_ind), model);

        ac = ac + sum(predict_label==trainLabel(test_ind));
    end
    
    ac = ac / length(trainLabel);
    fprintf('Cross-validation Accuracy = %g%%\n', ac * 100);
    cv(i).acc=ac;
    cv(i).com=com;
    
    
    if (ac >= bestcv),
        bestcv = ac; bestCom = com;
    end
    fprintf('%g %g (best com=%g, rate=%g)\n', com, ac, bestCom, bestcv);
    i = i + 1;
end
    
    [ model ] = trainHMMGMM(trainDataCell, trainLabel, bestCom );

end
