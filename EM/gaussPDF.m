function prob = gaussPDF(Data, Mu, Sigma)
% 用于返回一个多维高斯分布样本点的概率
% Inputs -----------------------------------------------------------------
%   o Data:  D x N array representing N datapoints of D dimensions.
%   o Mu:    D x 1 vector representing the center of the Gaussian.
%   o Sigma: D x D array representing the covariance matrix of the Gaussian.
% Output -----------------------------------------------------------------
%   o prob:  1 x N vector representing the likelihood of the N datapoints.
% 参考文献：
% @article{Calinon15,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%   year="2015"

[nbVar,nbData] = size(Data);
Data = Data' - repmat(Mu',nbData,1);
prob = sum((Data/Sigma).*Data, 2);
prob = exp(-0.5*prob) / sqrt((2*pi)^nbVar * abs(det(Sigma)) + realmin);
