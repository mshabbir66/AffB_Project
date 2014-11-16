function labels = GenerateAffectBurstLabelsForSingleFile(AffectBursts,fileName,numberOfFrames)


tempAffects = AffectBursts(strcmp(extractfield(AffectBursts,'fileName'),fileName));
labels = zeros(1,numberOfFrames);
for j = 1:length(tempAffects)
    labels = labels | GenerateAffectBurstLabel(tempAffects(j),numberOfFrames);
end


end