%% ����75�����Ե�
Mu = [mu1; mu2]; 
Sigma = cat(3,sigma1,sigma2); 
p = [0.75 0.25]; 
gmTrue = gmdistribution(Mu,Sigma,p);%����һ����˹���ģ��
X0 = random(gmTrue,75);
% �����ݾ���
[idx0,~,P0] = cluster(gm,X0);

figure;
l = ezcontour(@(x,y)pdf(gm,[x y]),[min(X0(:,1)) max(X0(:,1))],...
    [min(X0(:,2)) max(X0(:,2))]);
hold on;
gscatter(X0(:,1),X0(:,2),idx0,'rb','+o');
legend('ͶӰ����','Cluster 1','Cluster 2','Location','NorthWest');
title('���������ݷ���Ч��')
hold off;
set(gcf, 'Position', [100 100 400 320]);
set(l, 'LineWidth', 2);