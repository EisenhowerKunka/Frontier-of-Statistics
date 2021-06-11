%% ����
close all
clear
mu1 = [1 2];
Sigma1 = [1 0; 0 1];
mu2 = [3 4];
Sigma2 = [0.5 0; 0 0.5];
X1 = [mvnrnd(mu1,Sigma1,100);mvnrnd(mu2,Sigma2,100)];
% ��������к�ǰ������������صģ�������׳��ֲ�̬�����
X = [X1,X1(:,1)+X1(:,2)];
 
rng(1); % Ϊ���ظ���fit GMM�ǳ�ʼֵ��ѡȡ�������
try
    gm = fitgmdist(X,2)
catch exception
    disp('���ʱ����������')
    error = exception.message
end
% ����������
gm = fitgmdist(X,2,'RegularizationValue',0.1)
% ����cluster��������
idx = cluster(gm,X);
estimated_label = idx;
ground_truth_label = [2*ones(100,1); ones(100,1)];
k = find(estimated_label ~= ground_truth_label);
% ��Ǵ������ĵ�Ϊ����3
idx(k,1) = 3;
cluster1 = idx == 1;
cluster2 = idx == 2;
cluster3 = idx == 3

% ��ͼ
subplot(121)
scatter3(X(1:100,1),X(1:100,2),X(1:100,3), 15, 'r',  'filled');
hold on
scatter3(X(101:end,1),X(101:end,2),X(101:end,3), 15, 'b',  'filled');
title('ԭʼ����')
legend('Model-0','Model-1', 'Location', 'SouthEast')
% set(gca, 'YLim', [-8 6], 'XLim', [-6 6], 'FontSize', 9);
set(gcf, 'Position', [100 100 400 300]);
hold off

subplot(122)
scatter3(X(cluster1,1),X(cluster1,2),X(cluster1,3), 15, 'b',  'filled');
hold on
scatter3(X(cluster2,1),X(cluster2,2),X(cluster2,3), 15, 'r',  'filled');
scatter3(X(cluster3,1),X(cluster3,2),X(cluster3,3), 20, 'g',  'filled');
title('������')
legend('Model-0','Model-1', 'error', 'Location', 'SouthEast')
set(gcf, 'Position', [100 100 800 300]);
hold off


%% ���GMMʱ��kѡ������
close all
clear
% ����pca����̽��
% �������ݼ���������ݼ���UCI��������Ϣ���Բ鿴UCI��վ
load fisheriris
classes = unique(species)
% meas����Ҫ�������ݣ�4ά
% ��pca�㷨��ԭʼ���ݽ�ά��score������ֵ�Ӵ�С���еĽ��
[~,score] = pca(meas,'NumComponents',2);
 
% �ֱ���ʹ�ò�ͬ��k���������
GMModels = cell(3,1); % �洢������ͬ��GMMģ��
% ��������������������
options = statset('MaxIter',1000);
rng(1); % For reproducibility

% ����ѡ��ͬ��components�����ģ��
for j = 1:3
    GMModels{j} = fitgmdist(score,j,'Options',options);
    fprintf('\n GM Mean for %i Component(s)\n',j)
    Mu = GMModels{j}.mu
end
 
figure
for j = 1:3
    subplot(2,2,j)
    % gscatter���Ը����飨Ҳ����label�����ֵĻ���ɢ��ͼ
    % ��������2ά����Ϣ�����ӻ�
    gscatter(score(:,1),score(:,2),species)
    h = gca;
    hold on
    ezcontour(@(x1,x2)pdf(GMModels{j},[x1 x2]),...
        [h.XLim h.YLim],100)
    title(sprintf('GMMģ�� (K = %i) ',j));
    xlabel('��һ����');
    ylabel('�ڶ�����');
    if(j ~= 3)
        legend off;
    end
    set(gca, 'FontSize', 10);
    hold off
end
g = legend;
g.Position = [0.7 0.25 0.1 0.1];
set(gcf, 'Position', [100 100 500 400]);

%% ��ϸ�˹���ģ��ʱ�����ó�ʼֵ
clear
close all
% �������ݼ�������ֻʹ�ú���������
load fisheriris
X = meas(:,3:4);

% ����Ĭ�ϵĳ�ʼֵ���һ��GMM,����K=3
rng(10); % For reproducibility
GMModel1 = fitgmdist(X,3);

% ���һ��GMM������ÿ��ѵ�������ı�ǩ
% y�е����ִ���ͬ������
y = ones(size(X,1),1);
y(strcmp(species,'setosa')) = 2;
y(strcmp(species,'virginica')) = 3;
% ���ģ��
GMModel2 = fitgmdist(X,3,'Start',y);

% ���һ��GMM�� ��ʽ��������ʼ��ֵ��Э����ͻ��ϵ��.
Mu = [1 1; 2 2; 3 3];       % ��ֵ
Sigma(:,:,1) = [1 1; 1 2];  % ÿ���ɷֵ�Э�������
Sigma(:,:,2) = 2*[1 1; 1 2];
Sigma(:,:,3) = 3*[1 1; 1 2];
PComponents = [1/2,1/4,1/4];% ���ϵ��
S = struct('mu',Mu,'Sigma',Sigma,'ComponentProportion',PComponents);
GMModel3 = fitgmdist(X,3,'Start',S);

% ����gscatter������ͼ
figure
subplot(2,2,1)
% ԭʼ����
h = gscatter(X(:,1),X(:,2),species,[],'o',4);
haxis = gca;
xlim = haxis.XLim;
ylim = haxis.YLim;
d = (max([xlim ylim])-min([xlim ylim]))/1000;
[X1Grid,X2Grid] = meshgrid(xlim(1):d:xlim(2),ylim(1):d:ylim(2));
hold on
% GMMģ������ͼ
contour(X1Grid,X2Grid,reshape(pdf(GMModel1,[X1Grid(:) X2Grid(:)]),...
    size(X1Grid,1),size(X1Grid,2)),20)
uistack(h,'top')
title('{\bf �����ʼֵ}');
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
title('{\bf ���ݱ�ǩȷ����ʼֵ}');
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
title('{\bf ������ʼֵ}');
xlabel('Sepal length');
ylabel('Sepal width');
legend('Location',[0.7,0.25,0.1,0.1]);
hold off

% ��ʾ����ģ�͵ľ�ֵ.
table(GMModel1.mu,GMModel2.mu,GMModel3.mu,'VariableNames',...
    {'Model1','Model2','Model3'})
