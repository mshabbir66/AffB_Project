clear all;clc;

load Dataset/visseq.mat

fileName = testFiles(j).name(1:end-4);
[y,fs] = wavread(['..\Session',fileName(5),'\dialog\wav\',fileName,'.wav']);
fidv = fopen(['../Session',fileName(5),'/dialog/MOCAP_rotated/',fileName,'.txt'],'r');

visSeqRaw.data=textscan(fidv,'%d %f %s','Delimiter','\n','Headerlines',2);
fclose(fidv);

visSeq(length(visSeqRaw),1).data = [];

visSeq.data = zeros(length(visSeqRaw.data{3}),165);
for i = 1:length(visSeqRaw.data{3})
    visSeq.data(i,:)  = str2double(strsplit(visSeqRaw.data{3}{i}));
end



W=[];
for i=1:length(visseq)
    
    for k=1:length(visseq(i).data{1,3})
        
        data=textscan(visseq(i).data{1,3}{k},'%f','Delimiter',' ');
        S=reshape(data{1,1},3,55);
        W=[W S'];
        
    end
    disp(['done with the sample ', num2str(i)]);
end

Mask =isnan(W);
Mask2 = sum(Mask,1)==0;
W = W(:,Mask2);
% W = W - min(W,[],2)*ones(1,size(W,2));
%
% W = W./(max(W,[],2)*ones(1,size(W,2)));
% W(isnan(W(:))) = 0;

%meanW = W;%W- ones(size(W,1),1)*mean(W,1);

pcaW = zeros(3*size(W,1),size(W,2)/3);

pcaW(1:3:end,:) = W(:,1:3:end);
pcaW(2:3:end,:) = W(:,2:3:end);
pcaW(3:3:end,:) = W(:,3:3:end);

pcaWmean= mean(pcaW,2);
pcaW= pcaW-pcaWmean*ones(1,size(pcaW,2));


[U,Sn,V] = svd(pcaW*pcaW');
%[U,Sn,V] = svd(pcaW(:,1:165));

%% test
K=160;
n=340;
S = pcaW(:,1:n);

S = S+ pcaWmean*ones(1,n);

C=ExtractPCA(S,U,pcaWmean,K);


% Ut=U(1:K,:);
% C=S'*Ut';
Ut=U(1:K,:);
recS = (C*Ut)';

diff=S-recS;

recErr=mean(diff.^2);

recAcc=recErr/mean(pcaWmean.^2);

recS = recS+pcaWmean*ones(1,n);
%S = S+ pcaWmean*ones(1,n);

close all
for i=1:n
    
    hold off
    plot3(S(1:3:end,i),S(2:3:end,i),S(3:3:end,i),'r.');
    axis equal
    hold on
    plot3(recS(1:3:end,i),recS(2:3:end,i),recS(3:3:end,i),'bO')
    pause(0.01)
end
%save('PCA_ses1234', 'U', 'pcaWmean');