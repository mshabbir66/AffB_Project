function [ac] = get_cv_ac_bin_probabilistic(y,x,param,nr_fold)
len=length(y);
ac = 0;
rand_ind = randperm(len);
for i=1:nr_fold % Cross training : folding
  test_ind=rand_ind([floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold)]');
  train_ind = [1:len]';
  train_ind(test_ind) = [];
  
  model = svmtrain(y(train_ind),x(train_ind,:),param);
  [pred,a,decv] = svmpredict(y(test_ind),x(test_ind,:),model,' -b 1');
  
  prob_values_ordered=zeros(size(decv));
  prob_values_ordered(:,model.Label)=decv;
 [val pred]=max(prob_values_ordered');
            
  ac = ac + sum(y(test_ind)==pred');
end
ac = ac / len;
fprintf('Cross-validation Accuracy = %g%%\n', ac * 100);
