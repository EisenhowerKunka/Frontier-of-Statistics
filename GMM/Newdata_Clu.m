%% 产生75个测试点
Mu = [mu1; mu2]; 
Sigma = cat(3,sigma1,sigma2); 
p = [0.75 0.25]; 
gmTrue = gmdistribution(Mu,Sigma,p);%生成一个高斯混合模型
X0 = random(gmTrue,75);
% 新数据聚类
[idx0,~,P0] = cluster(gm,X0);

figure;
l = ezcontour(@(x,y)pdf(gm,[x y]),[min(X0(:,1)) max(X0(:,1))],...
    [min(X0(:,2)) max(X0(:,2))]);
hold on;
gscatter(X0(:,1),X0(:,2),idx0,'rb','+o');
legend('投影轮廓','Cluster 1','Cluster 2','Location','NorthWest');
title('测试新数据分类效果')
hold off;
set(gcf, 'Position', [100 100 400 320]);
set(l, 'LineWidth', 2);