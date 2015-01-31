function labels = GenerateAffectBurstLabelsForSingleFile(AffectBursts,fileName,numberOfFrames,labelmap)
labels = zeros(1,numberOfFrames);
if (isempty(AffectBursts))
    return;
end
if nargin ==3
    tempAffects = AffectBursts(strcmp(extractfield(AffectBursts,'fileName'),fileName));
    for j = 1:length(tempAffects)
        temp = GenerateAffectBurstLabel(tempAffects(j),numberOfFrames);
        labels(labels==0) = temp(labels==0);
    end
else
    tempAffects = AffectBursts(strcmp(extractfield(AffectBursts,'fileName'),fileName));
    for j = 1:length(tempAffects)
        temp = GenerateAffectBurstLabel(tempAffects(j),numberOfFrames,labelmap);
        labels(labels==0) = temp(labels==0);
    end
end

end