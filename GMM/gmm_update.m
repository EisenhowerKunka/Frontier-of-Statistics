%% 正则化
close all
clear
mu1 = [1 2];
Sigma1 = [1 0; 0 1];
mu2 = [3 4];
Sigma2 = [0.5 0; 0 0.5];
X1 = [mvnrnd(mu1,Sigma1,100);mvnrnd(mu2,Sigma2,100)];
% 这里第三列和前两列是线性相关的，因此容易出现病态的情况
X = [X1,X1(:,1)+X1(:,2)];
 
rng(1); % 为了重复，fit GMM是初始值的选取是随机的
try
    gm = fitgmdist(X,2)
catch exception
    disp('拟合时出现了问题')
    error = exception.message
end
% 加入正则项
gm = fitgmdist(X,2,'RegularizationValue',0.1)
% 利用cluster方法聚类
idx = cluster(gm,X);
estimated_label = idx;
ground_truth_label = [2*ones(100,1); ones(100,1)];
k = find(estimated_label ~= ground_truth_label);
% 标记错误分类的点为数字3
idx(k,1) = 3;
cluster1 = idx == 1;
cluster2 = idx == 2;
cluster3 = idx == 3

% 绘图
subplot(121)
scatter3(X(1:100,1),X(1:100,2),X(1:100,3), 15, 'r',  'filled');
hold on
scatter3(X(101:end,1),X(101:end,2),X(101:end,3), 15, 'b',  'filled');
title('原始数据')
legend('Model-0','Model-1', 'Location', 'SouthEast')
% set(gca, 'YLim', [-8 6], 'XLim', [-6 6], 'FontSize', 9);
set(gcf, 'Position', [100 100 400 300]);
hold off

subplot(122)
scatter3(X(cluster1,1),X(cluster1,2),X(cluster1,3), 15, 'b',  'filled');
hold on
scatter3(X(cluster2,1),X(cluster2,2),X(cluster2,3), 15, 'r',  'filled');
scatter3(X(cluster3,1),X(cluster3,2),X(cluster3,3), 20, 'g',  'filled');
title('聚类结果')
legend('Model-0','Model-1', 'error', 'Location', 'SouthEast')
set(gcf, 'Position', [100 100 800 300]);
hold off


%% 拟合GMM时的k选择问题
close all
clear
% 利用pca数据探索
% 加载数据集，这个数据集在UCI，具体信息可以查看UCI网站
load fisheriris
classes = unique(species)
% meas是主要特征数据，4维
% 用pca算法对原始数据降维，score是特征值从大到小排列的结果
[~,score] = pca(meas,'NumComponents',2);
 
% 分别尝试使用不同的k来拟合数据
GMModels = cell(3,1); % 存储三个不同的GMM模型
% 参数声明，最大迭代次数
options = statset('MaxIter',1000);
rng(1); % For reproducibility

% 尝试选择不同的components来拟合模型
for j = 1:3
    GMModels{j} = fitgmdist(score,j,'Options',options);
    fprintf('\n GM Mean for %i Component(s)\n',j)
    Mu = GMModels{j}.mu
end
 
figure
for j = 1:3
    subplot(2,2,j)
    % gscatter可以根据组（也就是label）区分的画出散点图
    % 这里用了2维的信息，可视化
    gscatter(score(:,1),score(:,2),species)
    h = gca;
    hold on
    ezcontour(@(x1,x2)pdf(GMModels{j},[x1 x2]),...
        [h.XLim h.YLim],100)
    title(sprintf('GMM模型 (K = %i) ',j));
    xlabel('第一主轴');
    ylabel('第二主轴');
    if(j ~= 3)
        legend off;
    end
    set(gca, 'FontSize', 10);
    hold off
end
g = legend;
g.Position = [0.7 0.25 0.1 0.1];
set(gcf, 'Position', [100 100 500 400]);

%% 拟合高斯混合模型时，设置初始值
clear
close all
% 加载数据集，并且只使用后两个特征
load fisheriris
X = meas(:,3:4);

% 利用默认的初始值拟合一个GMM,声明K=3
rng(10); % For reproducibility
GMModel1 = fitgmdist(X,3);

% 拟合一个GMM，声明每个训练样本的标签
% y中的数字代表不同的种类
y = ones(size(X,1),1);
y(strcmp(species,'setosa')) = 2;
y(strcmp(species,'virginica')) = 3;
% 拟合模型
GMModel2 = fitgmdist(X,3,'Start',y);

% 拟合一个GMM， 显式的声明初始均值，协方差和混合系数.
Mu = [1 1; 2 2; 3 3];       % 均值
Sigma(:,:,1) = [1 1; 1 2];  % 每个成分的协方差矩阵
Sigma(:,:,2) = 2*[1 1; 1 2];
Sigma(:,:,3) = 3*[1 1; 1 2];
PComponents = [1/2,1/4,1/4];% 混合系数
S = struct('mu',Mu,'Sigma',Sigma,'ComponentProportion',PComponents);
GMModel3 = fitgmdist(X,3,'Start',S);

% 利用gscatter函数绘图
figure
subplot(2,2,1)
% 原始样本
h = gscatter(X(:,1),X(:,2),species,[],'o',4);
haxis = gca;
xlim = haxis.XLim;
ylim = haxis.YLim;
d = (max([xlim ylim])-min([xlim ylim]))/1000;
[X1Grid,X2Grid] = meshgrid(xlim(1):d:xlim(2),ylim(1):d:ylim(2));
hold on
% GMM模型轮廓图
contour(X1Grid,X2Grid,reshape(pdf(GMModel1,[X1Grid(:) X2Grid(:)]),...
    size(X1Grid,1),size(X1Grid,2)),20)
uistack(h,'top')
title('{\bf 随机初始值}');
xlabel('Sepal length');
ylabel('Sepal width');
legend off;
hold off
subplot(2,2,2)
h = gscatter(X(:,1),X(:,2),species,[],'o',4);
hold on
contour(X1Grid,X2Grid,reshape(pdf(GMModel2,[X1Grid(:) X2Grid(:)]),...
    size(X1Grid,1),size(X1Grid,2)),20)
uistack(h,'top')
title('{\bf 根据标签确定初始值}');
xlabel('Sepal length');
ylabel('Sepal width');
legend off
hold off
subplot(2,2,3)
h = gscatter(X(:,1),X(:,2),species,[],'o',4);
hold on
contour(X1Grid,X2Grid,reshape(pdf(GMModel3,[X1Grid(:) X2Grid(:)]),...
    size(X1Grid,1),size(X1Grid,2)),20)
uistack(h,'top')
title('{\bf 给定初始值}');
xlabel('Sepal length');
ylabel('Sepal width');
legend('Location',[0.7,0.25,0.1,0.1]);
hold off

% 显示估计模型的均值.
table(GMModel1.mu,GMModel2.mu,GMModel3.mu,'VariableNames',...
    {'Model1','Model2','Model3'})
