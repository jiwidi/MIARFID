datasets = {'expressions','gauss2D','gender','iris','news','OCR_14x14','videos'};
for dataset = 1:length(datasets)
    printf("Testing dataset %s \n",datasets{dataset})
    data_path = strcat("data/", datasets{dataset},".gz");
    load(data_path);
    [N,L]=size(data);
    D=L-1;
    ll=unique(data(:,L));
    C=numel(ll);
    rand("seed",23); data=data(randperm(N),:);
    M=N-round(.7*N);
    te=data(N-M+1:N,:);
    weights_path = strcat("weights/", datasets{dataset},"_w");
    load(weights_path); rl=zeros(M,1);
    for m=1:M
        tem=[1 te(m,1:D)]';
        rl(m)=ll(linmach(w,tem)); end
    [nerr m]=confus(te(:,L),rl); output_precision(2);
    m=nerr/M
    s=sqrt(m*(1-m)/M)
    r=1.96*s
    printf("I=[%.3f, %.3f]\n",m-r,m+r);
endfor