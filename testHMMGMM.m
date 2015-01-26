function [predict_label, accuracy, prob_values] = testHMMGMM(testLabel, testDataCell, model)
accuracy = 0;

for i =1: length(model)
logp(:,i) = -1*hmmLogprob(model(i), testDataCell);
end
prob_values = -logp;

[~,predict_label] = min(logp,[],2);

