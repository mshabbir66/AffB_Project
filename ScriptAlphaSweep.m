%% confusion matrix

filenames = dir('./EXPproper/*.mat');
jrr =1;

for jr = 9:12
    
    load(['./EXPproper/',filenames(jr).name]);
    
    ConfusionMatrices(101).confdata=zeros(NClass);
    k=0;
    
    prob3d=[];prob=[];
    for i=1:nfold
        prob3d=[prob3d;acc3D(i).prob_values];
        prob=[prob;acc(i).prob_values];
    end
    
    for alfa=0:0.01:1
        k=k+1;
        fused = decisionFuserModified( prob, prob3d, alfa);
        
        [val, ind]=max(fused,[],2);
        fusedLabel=ind;
        predictLabels = fusedLabel;
        testLabels = extractfield(acc, 'testLabel');
        for i =1:NClass
            for j = 1:NClass
                ConfusionMatrices(k).confdata(i,j) = sum(predictLabels(testLabels==i)==j);
            end
        end
        ave_acc(k)=sum(diag(ConfusionMatrices(k).confdata))/sum(sum(ConfusionMatrices(k).confdata));
   
    end
    [maxacc, maxind]=max(ave_acc);
    ConfusionMatrix=ConfusionMatrices(maxind).confdata;
    best_alpha = xax(maxind);
    
    ConfusionMatrixSensitivity = ConfusionMatrix./(sum(ConfusionMatrix,2)*ones(1,NClass));
    ConfusionMatrixPrecision = ConfusionMatrix./(ones(NClass,1)*sum(ConfusionMatrix,1));
    
    Precision = mean(diag(ConfusionMatrixPrecision));
    Sensitivity = mean(diag(ConfusionMatrixSensitivity));
    
    subplot(2,4,jrr)
    xax=0:0.01:1;
    plot(xax,ave_acc);
    title([' S=',num2str(noS),' M=',num2str(noM)]);
    xlabel(['Acc =' num2str(100*maxacc) '% Alpha='  num2str(xax(maxind))])
    
    
    %% plot and metrics
    subplot(2,4,jrr+4)
    bar3(ConfusionMatrix');
    ax = gca;
    set(ax,'XTickLabel',axlabels);
    set(ax,'YTickLabel',axlabels);
    xlabel('GT');
    ylabel('P');
   
    
    ave_acc=sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix));
    
    title([' Acc: ' num2str(100*ave_acc) '% P: ' num2str(100*mean(Precision)) '% R: ' num2str(100*mean(Sensitivity)) '%']);
    
    
    jrr =jrr +1;
    
end