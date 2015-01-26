function model = trainHMMGMM(trainDataCell, trainLabel,nstates,nmix)

d = 26;
%nmix    = 4; % must specify nmix

for i = 1:length(unique(trainLabel))
    model(i) = hmmFit(trainDataCell(trainLabel==i), nstates, 'mixGaussTied', 'verbose', true, ...
        'nRandomRestarts', 1, 'maxiter', 100, 'nmix', nmix); 
end
