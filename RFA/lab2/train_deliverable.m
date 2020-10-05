datasets = {'expressions','gauss2D','gender','iris','news','videos'};

printf("Training perceptron for dataset %s \n",datasets{1})
data_path = strcat("data/", datasets{1},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
[w,E,k]=perceptron(data,10000,1);
weights_path = strcat("deliverable/", datasets{1},"_w");
save_precision(4);
save(weights_path,"w");

printf("Training perceptron for dataset %s \n",datasets{2})
data_path = strcat("data/", datasets{2},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
[w,E,k]=perceptron(data,1000,1);
weights_path = strcat("deliverable/", datasets{2},"_w");
save_precision(4);
save(weights_path,"w");

printf("Training perceptron for dataset %s \n",datasets{3})
data_path = strcat("data/", datasets{3},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
[w,E,k]=perceptron(data,1000,1);
weights_path = strcat("deliverable/", datasets{3},"_w");
save_precision(4);
save(weights_path,"w");

printf("Training perceptron for dataset %s \n",datasets{4})
data_path = strcat("data/", datasets{4},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
[w,E,k]=perceptron(data,100,1);
weights_path = strcat("deliverable/", datasets{4},"_w");
save_precision(4);
save(weights_path,"w");


printf("Training perceptron for dataset %s \n",datasets{5})
data_path = strcat("data/", datasets{5},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
[w,E,k]=perceptron(data,1,10000);
weights_path = strcat("deliverable/", datasets{5},"_w");
save_precision(4);
save(weights_path,"w");


printf("Training perceptron for dataset %s \n",datasets{6})
data_path = strcat("data/", datasets{6},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
[w,E,k]=perceptron(data,100,1);
weights_path = strcat("deliverable/", datasets{6},"_w");
save_precision(4);
save(weights_path,"w");





