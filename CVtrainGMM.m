function [ model, bestCom, cv ] = CVtrainGMM(trainData, trainLabel, comRange, nfoldCV)
%UNTÝTLED2 Summary of this function goes here
%   Detailed explanation goes here
% % #######################
% % Parameter selection using 3-fold cross validation
% % #######################
NClass=length(unique(trainLabel));
bestcv = 0;
i =1;
for com = comRange(1):comRange(2):comRange(3),
    %         cv(i) = get_cv_ac_bin(trainLabel, trainData, cmd, nfoldCV);
    
    
    len=length(trainLabel);
    ac = 0;
    rand_ind = randperm(len);
    for j=1:nfoldCV % Cross training : folding
        test_ind=rand_ind([floor((j-1)*len/nfoldCV)+1:floor(j*len/nfoldCV)]');
        train_ind = [1:len]';
        train_ind(test_ind) = [];
        
        [ model ] = trainGMM(trainData(train_ind), trainLabel(train_ind), com );
        
        testData=trainData(test_ind);
        testLabel=trainLabel(test_ind);
        Pos=zeros(length(testData),NClass);
        for j=1:length(testData)
            for class=1:NClass
                [~,Pos(j,class)] = posterior(model(class).obj,testData(j).data3d);
            end
        end
        [v ix] = sort(Pos,2);
        pred=ix(:,1);
        ac = ac + sum(ix(:,1)==testLabel);
    end
    ac = ac / len;
    fprintf('Cross-validation Accuracy = %g%%\n', ac * 100);
    cv(i)=ac;
    
    
    
    if (cv(i) >= bestcv),
        bestcv = cv(i); bestCom = com;
    end
    fprintf('%g %g (best com=%g, rate=%g)\n', com, cv(i), bestCom, bestcv);
    i = i + 1;
end
    
    [ model ] = trainGMM(trainData, trainLabel, bestCom );

end

