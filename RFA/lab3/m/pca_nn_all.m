datasets = {'expressions','gauss2D','gender','iris','news','videos'};

for dataset = 1:length(datasets)
  #!/usr/bin/octave -qf
    printf("Training pca_nn for dataset %s \n",datasets{dataset})
    Tr = strcat("../data/", datasets{dataset},"Tr");
    Te = strcat("../data/", datasets{dataset},"Te");
    load(sprintf(Tr)); tr=data; [NTr,L]=size(tr); D=L-1;
    labs=unique(data(:,L)); C=numel(labs);
    load(sprintf(Te)); te=data; NTe=rows(te); clear data;
    S=cov(tr(:,1:D)); [eigval,eigvec]=eigdec(S);
    st=sum(eigval); M=1; s=eigval(M);
    while (s<.95*st); M=M+1; s+=eigval(M); end;
    A=eigvec(:,1:M); trr=tr(:,1:D)*A; ter=te(:,1:D)*A;
    recolabs=zeros(1,NTe);
    for i=1:NTe
    tei=ter(i,1:M)';
    nmin=1; min=inf;
    for n=1:NTr
        trn=trr(n,1:M)'; aux=tei-trn; d=aux'*aux;
        if (d<min) min=d; nmin=n; endif
    end
    recolabs(i)=tr(nmin,L);
    end
    [Nerr m]=confus(te(:,L),recolabs);
    printf("%d/%d %s %s %d %d %.1f\n",M,D,Tr,Te,Nerr,NTe,100.0*Nerr/NTe);
    m
endfor