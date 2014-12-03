A = load('D:\Joker\AffB_Project\EXP\RecognitionVis_3class_decision.mat', 'acc');
B = load('D:\Joker\AffB_Project\EXP\RecognitionSound_3class_decision.mat', 'acc');
probSound = [];
probVis= [];
for i =1:384
probSound = [probSound, A.acc(i).prob];
probVis = [probVis, B.acc(i).prob];


end
epsilon  =1e-13;

probSound(probSound==0) = epsilon;

probVis(probVis==0) = epsilon;

[ fused ] = decisionFuser( double(probSound), double(probVis), 0.5);

testLabels = extractfield(A.acc, 'testLabel');

[~,predictLabels]  = max(fused,[],1);
