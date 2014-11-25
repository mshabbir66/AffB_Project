function [ normed ] = sigmoid_norm( val, m, s)
%UNTÝTLED4 Summary of this function goes here
%   Detailed explanation goes here
normed=1./(ones(size(val))+ exp( -((val-m*ones(size(val)))./(2*s)+1) ) );

end

