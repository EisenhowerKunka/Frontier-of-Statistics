function prob = gaussPDF(Data, Mu, Sigma)
% ���ڷ���һ����ά��˹�ֲ�������ĸ���
% Inputs -----------------------------------------------------------------
%   o Data:  D x N array representing N datapoints of D dimensions.
%   o Mu:    D x 1 vector representing the center of the Gaussian.
%   o Sigma: D x D array representing the covariance matrix of the Gaussian.
% Output -----------------------------------------------------------------
%   o prob:  1 x N vector representing the likelihood of the N datapoints.
% �ο����ף�
% @article{Calinon15,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%   year="2015"

[nbVar,nbData] = size(Data);
Data = Data' - repmat(Mu',nbData,1);
prob = sum((Data/Sigma).*Data, 2);
prob = exp(-0.5*prob) / sqrt((2*pi)^nbVar * abs(det(Sigma)) + realmin);
