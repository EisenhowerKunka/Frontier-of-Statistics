%% fisher
clc
clear
close all
load fisheriris
scatter(meas(1:50,1), meas(1:50,2), 15, 'r', 'filled')
hold on
scatter(meas(51:100,1), meas(51:100,2), 15, 'g', 'filled')
scatter(meas(101:end,1), meas(101:end,2), 15, 'b', 'filled')
xlabel('Sepal length (cm)');ylabel('Sepal width (cm)')
f = gcf
set(f, 'Position', [100 100 300 240])
a = gca
title('Fisher Iris Dataset');
set(a, 'FontSize', 9)
box on;

%% ���ɸ�˹���ģ��
% ����GMM��Ҫ�Ĳ���
clc
clear
close all
Mu = [1 2;-3 -5];                    % ��ֵ
Sigma = cat(3,[2 0;0 .5],[1 0;0 1]); % ���cat����������������ĳ��ά�Ͻ�������
P = ones(1,2)/2;                     % ���ϵ��

% ����GMMģ��
gm = gmdistribution(Mu,Sigma,P);   
% ��ʾGMM������
properties = properties(gm)
 
% ͼʾGMM��PDF
gmPDF = @(x,y)pdf(gm,[x y]); 
 
f = figure
set(f, 'Position', [100 100 800 400]);
p1 = subplot(121);
ezsurf(gmPDF,[-10 10],[-10 10])
title('PDF of the GMM');
set(p1, 'FontSize', 9)
% ͼʾCDF
gmCDF = @(x,y)cdf(gm,[x y]); 
 
p2 = subplot(122);
ezsurf(@(x,y)cdf(gm,[x y]),[-10 10],[-10 10])
title('CDF of the GMM');
set(p2, 'FontSize', 9)


%% ���һ��GMMģ��
% ����������ά�ĵ���˹ģ�ͣ�����������ģ������
close all
clear
% ��һ����˹
mu1 = [1 2];
Sigma1 = [2 0; 0 0.5];
% �ڶ�����˹
mu2 = [-3 -5];
Sigma2 = [1 0;0 1];
rng(1); % Ϊ���ظ�����
% ����������˹ģ�ͣ��ֱ��������1000�������㣬�������һ��
X = [mvnrnd(mu1,Sigma1,1000);mvnrnd(mu2,Sigma2,1000)]; 

% ģ����ϣ�����2���ɷ֣�gm��һ���ṹ�壬���������ģ�͵Ĳ���
gm = fitgmdist(X, 2);

% ������ϵĸ�˹ģ��
y = [zeros(1000,1);ones(1000,1)];   % �������ݵı�ǩ
h = gscatter(X(:,1),X(:,2),y);
% set(gca, 'YLim', [-10 10]);
hold on
ezcontour(@(x1,x2)pdf(gm,[x1 x2]),get(gca,{'XLim','YLim'}))
title('{\bf ɢ��ͼ����ϵĸ�˹ģ������}')
legend(h,'Model-0','Model-1', 'Location', 'SouthEast')
set(gca, 'YLim', [-8 6], 'XLim', [-6 6], 'FontSize', 9);
set(gcf, 'Position', [100 100 400 300]);
hold off

% ��ӡ����
properties(gm)
gm.mu
gm.Sigma

%% ����GMM����
% ��������
clear
close all
rng default;  % For reproducibility
mu1 = [1 2];
sigma1 = [3 .2; .2 2];
mu2 = [-1 -2];
sigma2 = [2 0; 0 1];
%2����˹����������
X = [mvnrnd(mu1,sigma1,200); mvnrnd(mu2,sigma2,100)]; 
n = size(X,1);
 
scatter(X(1:200,1),X(1:200,2),15,'ro','filled');
hold on; 
scatter(X(201:end,1),X(201:end,2),15,'bo','filled');
set(gcf, 'Position', [100 100 450 360]);
title('��������');
legend('cluster-1', 'cluster-2', 'Location', 'SouthEast');
set(gca, 'FontSize', 10);

% ���ģ��
options = statset('Display','final');
gm = fitgmdist(X,2,'Options',options)
% �������ģ�͵�ͶӰɢ��ͼ��
hold on
ezcontour(@(x,y)pdf(gm,[x y]),[-6 6],[-6 6]);
title('ɢ��ͼ�����GMMģ��')
xlabel('x'); ylabel('y');
set(gcf, 'Position', [100 100 450 360]);

% ����cluster��������
idx = cluster(gm,X);
estimated_label = idx;
ground_truth_label = [ones(200,1); 2*ones(100,1)];
k = find(estimated_label ~= ground_truth_label);
% ��Ǵ������ĵ�Ϊ����3
idx(k,1) = 3;

figure;
gscatter(X(:,1),X(:,2),idx);
legend('Cluster 1','Cluster 2','error', 'Location','NorthWest');
title('GMM����');
set(gcf, 'Position', [100 100 400 320]);

% ����������
% p ��n*2����ÿһ����һ�������㣬ÿһ�д������������������ȴ�С
P = posterior(gm,X);
% ������
cluster1 = (idx == 1); 
cluster2 = (idx == 2); 
figure;
% ���1
scatter(X(cluster1,1),X(cluster1,2),15,P(cluster1,1),'+')
hold on
scatter(X(cluster2,1),X(cluster2,2),15,P(cluster2,1),'o')
hold off
clrmap = jet(80);
colormap(clrmap(9:72,:))
ylabel(colorbar,'�������1�ĺ������')
title('�������1�ĺ������')
legend('cluster-1', 'cluster-2')
set(gcf, 'Position', [100 100 400 320]);
box on

%% Ԥ��
% �˹�����75�����Ե�
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

%% ����������  
clear; close all
rng(3)  % For reproducibility
mu1 = [1 2];
sigma1 = [3 .2; .2 2];
mu2 = [-1 -2];
sigma2 = [2 0; 0 1];
% �����������
X = [mvnrnd(mu1,sigma1,200); mvnrnd(mu2,sigma2,100)];
 
gm = fitgmdist(X,2);
% ��������������[.4, .6]��Χ�ڣ�����Ϊ����ͬʱ
threshold = [0.4 0.6];
% ��posterior��������������X����ÿ���ɷֵĺ�����ʣ�p��n*k����
P = posterior(gm,X);
% n�����������������sort������ÿ����������ȴ�С��������ֻ��������
n = size(X,1);
% order����������ֵ��С����Ķ�Ӧ����������
[~,order] = sort(P(:,1));
figure
subplot(121)
plot(1:n,P(order,1),'r-',1:n,P(order,2),'b-', 'LineWidth', 1.5)
legend({'Cluster 1', 'Cluster 2'})
ylabel('������')
xlabel('������')
title('GMM���������������')
% ȷ��ͬʱ����������ĵ�
idx = cluster(gm,X);
idxBoth = find(P(:,1)>=threshold(1) & P(:,1)<=threshold(2)); 
% ����ͬʱ��������cluster����������
numInBoth = numel(idxBoth)
 
subplot(122)
gscatter(X(:,1),X(:,2),idx,'rb','po',5)
hold on
scatter(X(idxBoth,1),X(idxBoth,2), 30, 'b','filled')
legend({'Cluster 1','Cluster 2','Both Clusters'},'Location','SouthEast', 'FontSize', 8)
title('�����')
xlabel('$x$', 'Interpreter', 'Latex')
ylabel('$y$', 'Interpreter', 'Latex')
hold off
set(gcf, 'Position', [100 100 600 260]);


