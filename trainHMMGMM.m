function model = trainHMMGMM(trainData, trainLabel)

d = 26;
nstates = 6;
nmix    = 6; % must specify nmix


trainDataCell = struct2cell(trainData);
%select sound
trainDataCell2 = trainDataCell(2,:)';
trainDataCell = trainDataCell(1,:)';

for i = 1:length(trainDataCell)
    trainDataCell{i} =[trainDataCell{i}';trainDataCell2{i}'];
end


for i = 1:length(unique(trainLabel))
    model(i) = hmmFit(trainDataCell(trainLabel==i), nstates, 'mixGaussTied', 'verbose', true, ...
        'nRandomRestarts', 2, 'maxiter', 100, 'nmix', nmix); 
end
