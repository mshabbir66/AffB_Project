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
