datasets = {'expressions','gauss2D','gender','iris','news','videos'};


printf("Testing dataset %s \n",datasets{1})
data_path = strcat("data/", datasets{1},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
M=N-round(.7*N);
te=data(N-M+1:N,:);
weights_path = strcat("weights/", datasets{1},"-a",num2str(1),"-b",num2str(10000),"_w");
load(weights_path); rl=zeros(M,1);
for m=1:M
    tem=[1 te(m,1:D)]';
    rl(m)=ll(linmach(w,tem)); end
[nerr m]=confus(te(:,L),rl); output_precision(2);
m=nerr/M
s=sqrt(m*(1-m)/M)
r=1.96*s
printf("I=[%.3f, %.3f]\n",m-r,m+r);

printf("Testing dataset %s \n",datasets{2})
data_path = strcat("data/", datasets{2},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
M=N-round(.7*N);
te=data(N-M+1:N,:);
weights_path = strcat("weights/", datasets{2},"-a",num2str(1),"-b",num2str(100),"_w");
load(weights_path); rl=zeros(M,1);
for m=1:M
    tem=[1 te(m,1:D)]';
    rl(m)=ll(linmach(w,tem)); end
[nerr m]=confus(te(:,L),rl); output_precision(2);
m=nerr/M
s=sqrt(m*(1-m)/M)
r=1.96*s
printf("I=[%.3f, %.3f]\n",m-r,m+r);

printf("Testing dataset %s \n",datasets{3})
data_path = strcat("data/", datasets{3},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
M=N-round(.7*N);
te=data(N-M+1:N,:);
weights_path = strcat("weights/", datasets{3},"-a",num2str(1),"-b",num2str(1000),"_w");
load(weights_path); rl=zeros(M,1);
for m=1:M
    tem=[1 te(m,1:D)]';
    rl(m)=ll(linmach(w,tem)); end
[nerr m]=confus(te(:,L),rl); output_precision(2);
m=nerr/M
s=sqrt(m*(1-m)/M)
r=1.96*s
printf("I=[%.3f, %.3f]\n",m-r,m+r);

printf("Testing dataset %s \n",datasets{4})
data_path = strcat("data/", datasets{4},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
M=N-round(.7*N);
te=data(N-M+1:N,:);
weights_path = strcat("weights/", datasets{4},"-a",num2str(100),"-b",num2str(1),"_w");
load(weights_path); rl=zeros(M,1);
for m=1:M
    tem=[1 te(m,1:D)]';
    rl(m)=ll(linmach(w,tem)); end
[nerr m]=confus(te(:,L),rl); output_precision(2);
m=nerr/M
s=sqrt(m*(1-m)/M)
r=1.96*s
printf("I=[%.3f, %.3f]\n",m-r,m+r);

printf("Testing dataset %s \n",datasets{5})
data_path = strcat("data/", datasets{5},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
M=N-round(.7*N);
te=data(N-M+1:N,:);
weights_path = strcat("weights/", datasets{5},"-a",num2str(1),"-b",num2str(10000),"_w");
load(weights_path); rl=zeros(M,1);
for m=1:M
    tem=[1 te(m,1:D)]';
    rl(m)=ll(linmach(w,tem)); end
[nerr m]=confus(te(:,L),rl); output_precision(2);
m=nerr/M
s=sqrt(m*(1-m)/M)
r=1.96*s
printf("I=[%.3f, %.3f]\n",m-r,m+r);

printf("Testing dataset %s \n",datasets{6})
data_path = strcat("data/", datasets{6},".gz");
load(data_path);
[N,L]=size(data);
D=L-1;
ll=unique(data(:,L));
C=numel(ll);
rand("seed",23); data=data(randperm(N),:);
M=N-round(.7*N);
te=data(N-M+1:N,:);
weights_path = strcat("weights/", datasets{6},"-a",num2str(100),"-b",num2str(1),"_w");
load(weights_path); rl=zeros(M,1);
for m=1:M
    tem=[1 te(m,1:D)]';
    rl(m)=ll(linmach(w,tem)); end
[nerr m]=confus(te(:,L),rl); output_precision(2);
m=nerr/M
s=sqrt(m*(1-m)/M)
r=1.96*s
printf("I=[%.3f, %.3f]\n",m-r,m+r);