datasets = {'expressions','gauss2D','gender','iris','news','OCR_14x14','videos'};

function [E] = test_dataset_parameters(dataset,a,b)
    % dataset="expressions";
    printf("Testing dataset %s \n",dataset)
    data_path = strcat("data/", dataset,".gz");
    load(data_path);
    [N,L]=size(data);
    D=L-1;
    ll=unique(data(:,L));
    C=numel(ll);
    rand("seed",23); data=data(randperm(N),:);
    M=N-round(.7*N);
    te=data(N-M+1:N,:);
    weights_path = strcat("weights/", dataset,"-a",num2str(a),"-b",num2str(b),"_w");
    load(weights_path); rl=zeros(M,1);
    for m=1:M
        tem=[1 te(m,1:D)]';
        rl(m)=ll(linmach(w,tem)); end
    [nerr m]=confus(te(:,L),rl); output_precision(2);
    m=nerr/M
    s=sqrt(m*(1-m)/M)
    r=1.96*s
    printf("I=[%.3f, %.3f]\n",m-r,m+r);
    E = m;
    IL = m-r;
    IH = m+r;
endfunction

% for dataset = 1:length(datasets)
    % for a=[.1 1 10 100 1000 10000 100000]
    %     for b=[.1 1 10 100 1000 10000 100000]
    %         [E,IL,IH]=test_dataset_parameters(datasets{dataset},b,a);
    %     endfor
    % endfor
% endfor


for dataset = 7:7 %length(datasets)
    list_a=[10 10 10 10 10 10 10];
    list_b=[.1 1 10 100 1000 10000 100000];
    V=zeros(7,7);
    V
    i_a = 1;
    i_b = 1;
    for a=[10 10 10 10 10 10 10]
        i_b = 1;
        for b=[.1 1 10 100 1000 10000 100000]
            E=test_dataset_parameters(datasets{dataset},a,b);
            V(i_a,i_b) = E;
            i_a
            i_b
            i_b = i_b +1;
        endfor
        i_a = i_a +1;
    endfor
    SurfObj = surf(list_a,list_b,V) % use logarithmic axes
    Axes = get( SurfObj, 'parent' );
    set( Axes, 'xscale', 'log', 'yscale', 'log' )   % use logarithmic axes
    xlabel("Alpha");
    ylabel("Beta")
    title ("Test error for videos dataset across diferent A&B values")
    print -djpg errorplot/videos.jpg
endfor