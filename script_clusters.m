clear all;
close all;
clc;
disp('Please browse for Training file');
[file_tr,path_tr] = uigetfile('*.txt');
prompt='Please enter number of clusters (k value)';
k = input(prompt);
prompt2='Please enter number of iterations needed';
iter= input(prompt2);
prompt3='Please enter number of Principal Components needed';
pc= input(prompt3);
[class_accuracy]=PCA_Cluster(file_tr,k,iter,pc);
class2=[];
train=importdata(file_tr,' ');
for pc_cnt=1:size(train,2)-2
    for clust_cnt=1:10
        [class_accuracy1]=PCA_Clusters(file_tr,clust_cnt,iter,pc_cnt);
        class2(pc_cnt,clust_cnt)=class_accuracy1(end);
    end
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Plot%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
figure(1);
hold on;
plot(0:1:iter,class_accuracy);
caption=sprintf('Iteration vs Classification Accuracy \n Clusters = %d, Principal Components = %d', k, pc);
title(caption);
xlabel('Iteration') 
ylabel('Classification Accuracy in percentage') 
grid on;
hold off;

%%%%%
figure(2);
hold on;
plot(1:1:size(train,2)-2,class2,'-o');
legendCell = cellstr(num2str((1:1:10)', 'No: of Clusters=%-d'));
legend(legendCell,'Location','northwest','Orientation','vertical');
caption=sprintf('Number of principal components vs Classification Accuracy \n Number of Clusters 1 to 10');
title(caption);
xlabel('Number of Principal Components ') 
ylabel('Classification Accuracy in percentage') 
grid minor;
hold off;