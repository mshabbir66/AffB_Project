function PCAcoef = ExtractPCA(visSignal,U,pcaWmean,K)

S=visSignal-pcaWmean*ones(1,size(visSignal,2));

Ut=U(1:K,:);

for i=1:size(visSignal,2)
    In=isnan(S(:,i));
    Stemp=S(~In,i);
    Utemp=Ut(:,~In);
    PCAcoefframe=Stemp'*Ut';
    PCAcoef(i,:)=PCAcoefframe;
end

        

end