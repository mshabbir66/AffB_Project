function labels = GenerateAffectBurstLabel(AffectBurst,numberOfFrames)

soundFrameDuration = 1; %in ms

time = (0:soundFrameDuration:(numberOfFrames-1)*soundFrameDuration)';
labels = zeros(1,numberOfFrames);


startTimes = AffectBurst.startTime;
endTimes = AffectBurst.endTime;

labels(time>startTimes & time< endTimes) = 1;

end
