% %Matlab �Դ�K��ֵ�㷨����kmeansʵ��
clc;
clear;
close all;
X = [randn(100,2)*0.75+ones(100,2);randn(100,2)*0.5-ones(100,2)]; %���������������
% ��ʱ��XΪ200*2�ľ���

[idx,C] = kmeans(X,2,'Distance','cityblock','Replicates',5);%����K��ֵ�㷨���з���
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12) %���Ʒ�����һ�������
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12) %���Ʒ����ڶ��������
plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3) %���Ƶ�һ��͵ڶ������ݵ����ĵ�
legend('Cluster 1','Cluster 2','Centroids','Location','NW') 
title 'Cluster Assignments and Centroids'
hold off
