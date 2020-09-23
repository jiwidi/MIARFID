datasets = {'expressions','gauss2D','gender','iris','news','OCR_14x14','videos'};
for dataset = 1:length(datasets)
    for a=[.1 1 10 100 1000 10000 100000]
        for b=[.1 1 10 100 1000 10000 100000]
            printf("Training perceptron for dataset %s - alpha %f - beta %f \n",datasets{dataset},a,b)
            data_path = strcat("data/", datasets{dataset},".gz");
            load(data_path);
            [N,L]=size(data);
            D=L-1;
            ll=unique(data(:,L));
            C=numel(ll);
            rand("seed",23); data=data(randperm(N),:);
            [w,E,k]=perceptron(data(1:round(.7*N),:),b,a);
            weights_path = strcat("weights/", datasets{dataset},"-a",num2str(a),"-b",num2str(b),"_w");
            save_precision(4);
            save(weights_path,"w");
            % output_precision(2); w
        endfor
    endfor
endfor