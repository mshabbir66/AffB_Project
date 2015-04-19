function soundseq=CreateAudioMat(Samples)

soundseq(length(Samples)).data = [];

for i = 1:length(Samples)
    [y,fs] = wavread(['..\Session',Samples(i).fileName(5),'\dialog\wav\',Samples(i).fileName,'.wav']);
    startFrame = round(fs*(Samples(i).startTime)/1000);
    if startFrame < 1
        startFrame = 1;
    end
    endFrame = round(fs*Samples(i).endTime/1000);
    switch str2num(Samples(i).fileName(5))
        case 1
            y = y(startFrame:endFrame,1);
        case 2
            y = y(startFrame:endFrame,2);
        case 3
            y = y(startFrame:endFrame,2);
        case 4
            y = y(startFrame:endFrame,2);
    end  
    soundseq(i).data = y';
    disp(['done with ', num2str(i)]);
    
end

end
%save Dataset/soundseq.mat soundseq