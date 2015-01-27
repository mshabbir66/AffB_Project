
mlist=[2,4,8,16];
figure;
for j=1:4
    load(['EXPproper/GMMRecognitionDecFusedmix' num2str(mlist(j))],'ConfusionMatrix','xax','maxind','ave_acc');
    for i=1:2
        subplot(2,4,(i-1)*4+j);
        plot(xax,ave_acc);title(['#mix ' num2str(mlist(j)) ' max acc ' num2str(100*maxacc) '% for alpha='  num2str(xax(maxind))]);
        xlabel('alpha');ylabel('accuracy');
       if(i==2)
           
           bar3(ConfusionMatrix');
           ax = gca;
           set(ax,'XTickLabel',axlabels);
           set(ax,'YTickLabel',axlabels);
           xlabel('GT');
           ylabel('P');
           ConfusionMatrixSensitivity = ConfusionMatrix./(sum(ConfusionMatrix,2)*ones(1,NClass));
           ConfusionMatrixPrecision = ConfusionMatrix./(ones(NClass,1)*sum(ConfusionMatrix,1));

           Precision = mean(diag(ConfusionMatrixPrecision));
           Sensitivity = mean(diag(ConfusionMatrixSensitivity));
           
           title(['Acc: ' num2str(100*ave_acc(maxind)) '% Pre: ' num2str(100*mean(Precision)) '% Rec: ' num2str(100*mean(Sensitivity)) '%']);

       end
    end
end