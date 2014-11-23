function [ac] = get_cv_ac_bin_decisionBin(y,x,param,nr_fold)
len=length(y);
ac = 0;
rand_ind = randperm(len);
for i=1:nr_fold % Cross training : folding
  test_ind=rand_ind([floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold)]');
  train_ind = [1:len]';
  train_ind(test_ind) = [];
  %model = svmtrain(y(train_ind),x(train_ind,:),param);
  sherwood_train(x(train_ind,:)', y(train_ind), param);
  
  probabilities = sherwood_classify(x(test_ind,:)', param);
  [~,predictLabels] = max(probabilities,[],1);
  predictLabels = predictLabels';
  
  %predictLabels = ones(length(probabilities),1);
  %predictLabels(probabilities(1,:) < 0.5) = 2;
  
  %[pred,a,decv] = svmpredict(y(test_ind),x(test_ind,:),model);
  ac = ac + sum(y(test_ind)==predictLabels);
end
ac = ac / len;
fprintf('Cross-validation Accuracy = %g%%\n', ac * 100);