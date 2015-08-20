clc
clear all
close all


files=[];
for i=1:4
    file=dir(['../Session' num2str(i) '/dialog/wav']);
    file=file(3:end,:);
    files=[files;file];
end

testFiles = files;
count =0;

for j=1:length(testFiles)
    
    count=count+1;
    fileName = testFiles(j).name(1:end-4);
    [y,fs] = wavread(['..\Session',fileName(5),'\dialog\wav\',fileName,'.wav']);
    switch str2num(fileName(5))
        case 1
            y = y(:,1);
        case 2
            y = y(:,2);
        case 3
            y = y(:,2);
        case 4
            y = y(:,2);
        case 5
            y = y(:,2);
    end
    
    
    fidv = fopen(['../Session',fileName(5),'/dialog/MOCAP_rotated/',fileName,'.txt'],'r');
    
    visSeqRaw.data=textscan(fidv,'%d %f %s','Delimiter','\n','Headerlines',2);
    fclose(fidv);
    

    
    visSeq(length(visSeqRaw),1).data = [];
    
    visSeq.data = zeros(length(visSeqRaw.data{3}),165);
    for i = 1:length(visSeqRaw.data{3})
        visSeq.data(i,:)  = str2double(strsplit(visSeqRaw.data{3}{i}));
    end
    
    numberOfFrames=length(y)*1000/fs;
    
    unseenMocap = [];
    unseenMFCC = [];
    i=0;
    flg = (winSize+ winShift*i < length(y));
    flg = flg & (floor((winSize+winShift*i)/fs*vfs) <  size(visSeq.data,1));
    
    while flg
        %unseenMFCC(end+1,1).data = ExtractMFCC(y(1+winShift*i:winSize+ winShift*i),fs);
        startFrame  = floor((winShift*i/fs)*vfs)+1;
        endFrame= floor((winSize+winShift*i)/fs*vfs);
        unseenMocap(end+1,1).data = visSeq.data(startFrame:endFrame,:);
        unseenMFCC(end+1,1).data = ExtractMFCC(y(1+winShift*i:winSize+ winShift*i),fs);
        i = i + 1;
        flg = winSize+ winShift*i < length(y);
        flg = flg & (floor((winSize+winShift*i)/fs*vfs) <  size(visSeq.data,1));
        totalTime = (winSize+winShift*i)/fs;
    end
    clip(count).dataMocap=unseenMocap;
    clip(count).dataMFCC = unseenMFCC;
    
    clip(count).fileName=fileName;
    
end

