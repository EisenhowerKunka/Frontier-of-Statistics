%%���ر�Ҷ˹�㷨Matlabʵ��
clear all;
close all;
clc;
load fisheriris
X = meas;
Y = species;
Mdl = fitcnb(X,Y)  %%ѵ�����ر�Ҷ˹ģ��
Mdl.ClassNames  %%��ģ���еģ��������ƣ�����������ʾ�鿴
Mdl.Prior        %%��ģ���еģ�������ʣ�����������ʾ�鿴

%% ����ѵ���õ�ģ�ͣ�Ԥ��������������
predict(Mdl,[1])
