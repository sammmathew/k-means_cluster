function [class_accuracy]=PCA_Cluster(file_tr,k,iter,pc)
% k=5;
% iter=100;
% pc=3;
%%%%%%

train=importdata(file_tr,' '); %Input training file

x=(train(:,1:16)./max(train(:,1:16)));

%zero mean unit variance
mean_arr_X=mean((x));
sd_arr_X=std((x));
X=x-mean_arr_X;
XX=X./sd_arr_X;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%PCA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cov_mat=cov(XX);%COV matrix
[eig_vec,eig_val]=eig(cov_mat);%Eig val and Eig vector calculation

[d,ind] = sort(diag(eig_val),'descend');
eig_vector = eig_vec(:,ind);

pc_eig_vec=eig_vector(:,1:pc);%Principle Components

PC_points=pc_eig_vec'*XX';  %Projection of data point on Principal components
PC_points=PC_points';
PC=PC_points;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Clustering%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

it=0;
%iter=100
%
PC_all={};
PC_mean_all={};

% rng(1) is to ensure random number generated to have all of the 10 class
% is randomly generated once- Please remove rng(1) to see how program work
% differently
% class
%rng(1);
PC_points(:,end+1)=train(:,end);
PC_points(:,end+1)=randi([1 k],size(PC_points,1),1);%Initilaize each point to a random cluster
PC_points(1:k,end)=(1:1:k)';
classification_accuracy=[];
x(:,end+1)=PC_points(:,end);



%%%Iterations
while it<=iter
PC_points_sort=sortrows(PC_points,size(PC_points,2));%Sort the rows based on cluster index
x_sort=sortrows(x,size(PC_points,2));
%Mean calculation
PC_mean=[];
for i =1:k
    PC_mean(i,:)=mean(PC_points_sort(PC_points_sort(:,end)==i,1:pc));%Calulating mean of each cluster
end

%Cacluate distance from each point to cluster centroid
PC_mean_modify=repmat(permute(PC_mean,[3,2,1]),size(PC_points_sort,1),1);
d=PC-PC_mean_modify;
% di=d.^2;
% dis=sum(di,2);

dist_un=sqrt(sum((PC-PC_mean_modify).^2,2));
dist=permute(dist_un,[1,3,2]);

[M,I]=min(dist,[],2);%Changing cluster index to nearest cluster centroid
%PC_points_sort2=PC_points_sort;
PC_points(:,pc+2)=I;


%%
%%%Calculating classification accuracy

PC_all{it+1}=PC_points_sort;
PC_mean_all{it+1}=PC_mean;
classification_accuracy(it+1)=0;
class_accuracy(it+1)=0;
    for j=1:k
       cl=PC_points_sort(PC_points_sort(:,end)==j,end-1);
       [freq,fnum]=mode(cl);
       classification_accuracy(it+1)=classification_accuracy(it+1)+(fnum);
    end
class_accuracy(it+1)=(classification_accuracy(it+1)./size(x,1))*100;
it=it+1;
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Plot%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% %%%For plotting scatter plot in 3D and PC loadings-> applicable only if principal component chosen is 3

if(pc==3)
figure(3);
scatter3(PC_points(1:size(x,1),1),PC_points(1:size(x,1),2),PC_points(1:size(x,1),3));
hold
 plotv(pc_eig_vec(1:size(x,2)-1,:)','-')
quiver3(zeros(size(x,2)-1,1),zeros(size(x,2)-1,1),zeros(size(x,2)-1,1),pc_eig_vec(1:size(x,2)-1,1),pc_eig_vec(1:size(x,2)-1,2),pc_eig_vec(1:size(x,2)-1,3),6);
end

if(pc==2)
figure(3);
scatter(PC_points(1:size(x,1),1),PC_points(1:size(x,1),2));
hold
grid on;
 plot(pc_eig_vec(1:size(x,2)-1,:)','-')
quiver(zeros(size(x,2)-1,1),zeros(size(x,2)-1,1),pc_eig_vec(1:size(x,2)-1,1),pc_eig_vec(1:size(x,2)-1,2),6);
%plotv([pc_eig_vec(1:size(x,2)-1,1);pc_eig_vec(1:size(x,2)-1,2)]);
end
% 
% 