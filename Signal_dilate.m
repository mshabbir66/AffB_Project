function [ predictLabels ] = Signal_dilate( predictLabels, numberOfTimes )
%Signal_dilate predictLabels 1xn
%   Detailed explanation goes here
    laughterMask= predictLabels==1;
    %breathingMask = predictLabels==2;
    
    for i = 1:numberOfTimes
    laughterMask = laughterMask | [0,laughterMask(1:end-1)] | [laughterMask(2:end),0];
    %breathingMask = breathingMask | [0,breathingMask(1:end-1)] | [breathingMask(2:end),0];
    end
    
    predictLabels(laughterMask) = 1;
    %predictLabels(breathingMask) = 2;
    
    
end