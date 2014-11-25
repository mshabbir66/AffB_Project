function labels = GenerateAffectBurstLabel(AffectBurst,numberOfFrames,labelmap)

soundFrameDuration = 1; %in ms

time = (0:soundFrameDuration:(numberOfFrames-1)*soundFrameDuration)';
labels = zeros(1,numberOfFrames);


startTimes = AffectBurst.startTime;
endTimes = AffectBurst.endTime;
if nargin ==2
    labels(time>startTimes & time< endTimes) = 1;
else
    labels(time>startTimes & time< endTimes) = labelmap(AffectBurst.type);
end
end
