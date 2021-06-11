clear all,clc;
load fisheriris
t1=fitctree(meas,species,'PredictorNames',{'SL' 'SW' 'PL' 'PW'})
view(t1)
view(t1,'Mode','graph')

%% 下面开始减枝
t2=prune(t1,'level',2)
%t2=purne(t1,'nodes',nodes)
view(t2,'Mode','graph')

%%  输入测试数据
predict(t2,[1 0.2 0.4 2])