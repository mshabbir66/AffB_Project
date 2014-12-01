bar3(ConfusionMatrix);
ax = gca;
set(ax,'XTickLabel',{'Laughter','Breathing','Reject'});
set(ax,'YTickLabel',{'Laughter','Breathing','Reject'});
xlabel('GT');
ylabel('P');
title('Method: Decision Forests, Features: Sound + Visual (Decision Fusion)');

Precision = diag(ConfusionMatrixPrecision)
Sensitivity = diag(ConfusionMatrixSensitivity)
mean(Precision)
mean(Sensitivity)

ave_acc=100*sum(diag(ConfusionMatrix))/sum(sum(ConfusionMatrix));
title(['Confusion Matrix, ' ' Acc: ' num2str(ave_acc) '% Precision: ' num2str(100*mean(Precision)) '% Recall: ' num2str(100*mean(Sensitivity)) '%']);
