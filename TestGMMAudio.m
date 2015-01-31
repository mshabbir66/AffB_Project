function [prob_values,predict_label] = TestGMMAudio(testData,model,NClass)

    Pos=zeros(length(testData),NClass);
    for j=1:length(testData) 
        for class=1:NClass
            [~,Pos(j,class)] = posterior(model(class).obj,testData(j).data);           
        end
    end    
    [~, ix] = sort(Pos,2);
    predict_label = ix(:,1); 
    prob_values = Pos;
    
end