load Dataset/SILaughterRawData.mat

AffectDataSync=[];
addpath C:\Users\Berker\Documents\GitHub\SCE_project

IEMOCAP=[AffectBursts;antiAffectBursts';antiAffectBurstsforAVlaughterCycle'];

for i=1:length(IEMOCAP)
    IEMOCAP(i).sesNumber=str2num(IEMOCAP(i).fileName(5));
    IEMOCAP(i).gender=IEMOCAP(i).fileName(6);
end

temp=max(extractfield(IEMOCAP,'sesNumber'));
for i=1:length(AVlaughterCycleAffectBursts)
AVlaughterCycleAffectBursts(i).sesNumber=AVlaughterCycleAffectBursts(i).sesNumber+temp;
end

Combinedsoundseq=[Affsoundseq antiAffsoundseq antiAffsoundseqforAVlaughterCycle AVlaughterCycleAffsoundseq]';
Combined=[IEMOCAP;AVlaughterCycleAffectBursts'];

temp=extractfield(Combined,'sesNumber');
gender=extractfield(Combined,'gender');
for i=1:length(temp)
sesNum(i)=temp{i};
end

cnt=0;
genCode=['M','F'];
sessionList=unique(sesNum);
for k=sessionList
    for l=1:2
        mask =(sesNum==k)&strcmp(gender,genCode(l));
        if(sum(mask)==0)
            continue;
        end
        cnt=cnt+1;
        for i=1:length(mask)
            if(mask(i))
            Combined(i).speaker=cnt;
            end
        end
    end
end

tempFolderPath = './cmn/';

for i=1:max(unique(extractfield(Combined,'speaker')))
    for j=1:length(Combined)
        if(Combined(j).speaker==i)
           wavwrite(Combinedsoundseq(j).data,16000,['./cmn/temp' num2str(j) '.wav']);
        end
    end
    
    for j=1:length(Combined)
        if(Combined(j).speaker==i)
           [ out ] = ExtractAudioSamplesFromOneSeq ( Combinedsoundseq(j).data , j, Combined(j).type,num2str(Combined(j).fileName),Combined(j).sesNumber,Combined(j).gender,i,16000, 750, 250, 1 );
           AffectDataSync=[AffectDataSync;out];
        end
    end

    files = dir([tempFolderPath,'*.wav']);

    for k= 1:length(files)
        delete([tempFolderPath,files(k).name]);
    end
    
end