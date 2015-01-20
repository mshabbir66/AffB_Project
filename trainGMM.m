function [ model ] = trainGMM( trainData, trainLabel, NComponents )
%UNT�TLED2 Summary of this function goes here
%   Detailed explanation goes here
options = statset('MaxIter',1000);
Nclass=length(unique(trainLabel));
model(Nclass).obj=[];
    for class=1:Nclass
        data=[];
        for i=1:length(trainData)
            if(trainLabel(i)==class)
                data=[data;trainData(i).data3d];
            end
        end

        model(class).obj = gmdistribution.fit(data,NComponents,'Regularize',.1,'Options',options);
    end

end

