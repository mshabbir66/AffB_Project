function [ fused ] = decisionFuser( modality1, modality2, alfa1)
    modality1=log(modality1);
    modality2=log(modality2);
    
    mean1=mean(modality1(:));
    mean2=mean(modality2(:));
    
    std1=std(modality1(:));
    std2=std(modality2(:));
    
    normed1 = sigmoid_norm( modality1, mean1, std1);
    normed2 = sigmoid_norm( modality2, mean2, std2);
    
    fused=normed1*alfa1+normed2*(1-alfa1);
end