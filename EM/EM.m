clc
clear
close all
%-----------------------��������-------------------------------------------
mu1 = [1 2];
sigma1 = [3 .2; .2 2];
mu2 = [-1 -2];
sigma2 = [2 0; 0 1];
%2����˹����������
X = [mvnrnd(mu1,sigma1,200); mvnrnd(mu2,sigma2,100)]';  % ��ϵ�����
[nbVar, nbData] = size(X);   % ���ݵ�ά�Ⱥ͸���

% ����һ���ṹ�����ڱ���ģ�Ͳ���������
model.nbStates = 2; % ������������ȡֵ������GMM��˵����3���ɷ�
model.nbVar = nbVar;    % ���ݵ�ά��
model.nbData = nbData;
diagRegularizationFactor = 1E-4; % �������ѡ����

%-----------------------������ʼ��-------------------------------------------
% �����ݰ��մ�С�ֳ�nbStates���Σ�Ȼ����ÿ�η�Χ�ڵ����ݼ����ʼֵ
% �����ݰ���ĳ��ά�ȴ�������
[B, I] = sort(X(1,:));   % ���յ�һ�����򷵻�����
Data = X(:, I); % ����������
Sep = linspace(min(Data(1,:)), max(Data(1,:)), model.nbStates+1);
% �ֱ��ÿ���γ�ʼ��
for i=1:model.nbStates
	idtmp = find( Data(1,:)>=Sep(i) & Data(1,:)<Sep(i+1));  % �������ݶε�����
	model.Priors(i) = length(idtmp);    % ��ʼ����Ϊ���ݵ�ı���
	model.Mu(:,i) = mean(Data(:,idtmp)');   % ��ʼ����ֵ
	model.Sigma(:,:,i) = cov(Data(:,idtmp)');   % ��ʼ��Э�������
	%���򻯷�ֹЭ�����������ʽΪ0�����ּ���Ĳ��ȶ���
	model.Sigma(:,:,i) = model.Sigma(:,:,i) + eye(nbVar)*diagRegularizationFactor;
end
model.Priors = model.Priors / sum(model.Priors);

% EM�㷨�Ĳ���
nbMinSteps = 5; %Minimum number of iterations allowed
nbMaxSteps = 100; %����������
err_ll = 1E-6; % ��Ȼ�����ı仯�ʣ����仯С�������ֵʱ��˵��������


%-----------------------EM����-------------------------------------------
% ��ѭ����������ʼ
for nbIter=1:nbMaxSteps
	fprintf('.');
	
	%E-step�����������ʣ�L��ʾÿ��������ȡz=1,2...�ĸ���
    L = zeros(model.nbStates,size(Data,2)); % ��ʼ���������ڴ��w
    for i=1:model.nbStates  % ����ÿ��z
        L(i,:) = model.Priors(i) * gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i));
    end
    % sum(A, 1)������ͣ������������� repmat(A, m, n)����������ά���ϸ���A
    GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);   % �������
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);   % w_i/sum(w_i)
	
	%M-step
	for i=1:model.nbStates
		% ����phi������
		model.Priors(i) = sum(GAMMA(i,:)) / nbData;
		
		% ���¾�ֵ
		model.Mu(:,i) = Data * GAMMA2(i,:)';
		
		% ����Э�������
		DataTmp = Data - repmat(model.Mu(:,i),1,nbData);
		model.Sigma(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(size(Data,1)) * diagRegularizationFactor;
    end
	
    % ��ʾ��������
    if mod(nbIter , 4) == 0
        plot_em(nbIter, X', model.Mu, model.Sigma);
        pause(2);   % ��ͣ2��
    end
    
	% ������Ȼ����ֵ
	LL(nbIter) = sum(log(sum(L,1))) / nbData;
	%Stop the algorithm if EM converged (small change of LL)
	if nbIter>nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<err_ll || nbIter==nbMaxSteps-1
			disp(['EM�㷨�� ' num2str(nbIter) ' �ε���������.']);
			break;
		end
	end
end
if nbIter == nbMaxSteps-1
    disp(['�ﵽ��������������������������������������...']);
end

% -------------------------------�����õ�GMM�㷨�Ƚ�----------------------------
gm = fitgmdist(Data', 2);
plot_em(nbIter, X', model.Mu, model.Sigma);

% plot(Data(1,:),Data(2,:),'.','markersize',8,'color',[.7 .7 .7]);hold on;
hold on
plotGMM(gm.mu, gm.Sigma, [0 0.8 0], .5);

