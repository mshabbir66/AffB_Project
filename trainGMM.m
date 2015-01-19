function [ model ] = trainGMM( trainData, trainLabel, NComponents )
%UNTÝTLED2 Summary of this function goes here
%   Detailed explanation goes here
Nclass=length(unique(trainLabel));
model(Nclass).obj=[];
    for class=1:Nclass
        data=[];
        for i=1:length(trainData)
            if(trainLabel(i)==class)
                data=[data;trainData(i).data];
            end
        end

        model(class).obj = gmdistribution.fit(data,NComponents,'Regularize',.1);
    end

end

