

pred=extractfield(folds,'predict_label');
real=extractfield(folds,'testLabel');

audio=sum(pred==real)/length(real)


pred=extractfield(folds3d,'predict_label');
real=extractfield(folds3d,'testLabel');

video=sum(pred==real)/length(real)
