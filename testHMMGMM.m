function [predict_label, accuracy, prob_values] = testHMMGMM(testLabel, testData, model);

testDataCell = struct2cell(testData);
%select sound
testDataCell2 = testDataCell(2,:)';
testDataCell = testDataCell(1,:)';

for i = 1:length(testDataCell)
    testDataCell{i} =[testDataCell{i}';testDataCell2{i}'];
end

for i =1: length(model)
logp(:,i) = -1*hmmLogprob(model(i), testDataCell);
end

[~,predict_label] = min(logp,[],2);
