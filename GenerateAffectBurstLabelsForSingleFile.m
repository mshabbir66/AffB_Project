function labels = GenerateAffectBurstLabelsForSingleFile(AffectBursts,fileName,numberOfFrames,labelmap)
 if(isempty(AffectBursts))
     labels = zeros(1,numberOfFrames);
     return;
 end
if nargin ==3
    tempAffects = AffectBursts(strcmp(extractfield(AffectBursts,'fileName'),fileName));
    labels = zeros(1,numberOfFrames);
    for j = 1:length(tempAffects)
        temp = GenerateAffectBurstLabel(tempAffects(j),numberOfFrames);
        labels(labels==0) = temp(labels==0);
    end
else
    tempAffects = AffectBursts(strcmp(extractfield(AffectBursts,'fileName'),fileName));
    labels = zeros(1,numberOfFrames);
    for j = 1:length(tempAffects)
        temp = GenerateAffectBurstLabel(tempAffects(j),numberOfFrames,labelmap);
        labels(labels==0) = temp(labels==0);
    end
end

end