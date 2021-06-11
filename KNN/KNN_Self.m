clear all;
close all;
clc;
%%���ø�˹�ֲ������ɶ���Ƭ���ݺͱ�ǩ
aver1=[8 3];  %��ֵ
covar1=[2 0;0 2.5];  %2ά���ݵ�Э����
data1=mvnrnd(aver1,covar1,100);   %������˹�ֲ����ݣ�100��
for i=1:100    %���˹�ֲ����������еĸ���Ϊ0
    for j=1:2   %��Ϊ�򶷾�ͷ���ͽ��Ǿ�ͷ������Ϊ����
        if data1(i,j)<0
            data1(i,j)=0;
        end
    end
end
label1=ones(100,1);  %���������ݵı�ǩ����Ϊ1
plot(data1(:,1),data1(:,2),'+');  %��+���Ƴ�����
axis([-1 12 -1 12]); %�趨�������᷶Χ
xlabel('�򶷾�ͷ��'); %��Ǻ���Ϊ�򶷾�ͷ��
ylabel('���Ǿ�ͷ��'); %�������Ϊ���Ǿ�ͷ��
hold on;
%%���ø�˹�ֲ������ɰ���Ƭ���ݺͱ�ǩ
aver2=[3 8];
covar2=[2 0;0 2.5];
data2=mvnrnd(aver2,covar2,100); %������˹�ֲ�����
for i=1:100    %���˹�ֲ����������еĸ���Ϊ0
    for j=1:2  %��Ϊ�򶷾�ͷ���ͽ��Ǿ�ͷ������Ϊ����
        if data2(i,j)<0
            data2(i,j)=0;
        end
    end
end
plot(data2(:,1),data2(:,2),'ro');  %��o���Ƴ�����
label2=label1+1; %���������ݵı�ǩ����Ϊ2
data=[data1;data2];
label=[label1;label2];
K=11;   %�����࣬һ��Kȡ���������ڲ������������Ǹ���
%�������ݣ�KNN�㷨������������ĸ��࣬�������ݹ���25��
%�򶷾�ͷ������3-7�����Ǿ�ͷ��Ҳ����3-7
for movenum=3:1:7
    for kissnum=3:1:7
        test_data=[movenum kissnum];  %�������ݣ�Ϊ5X5����
        %%���濪ʼKNN�㷨����Ȼ������11NN��
        %��������ݺ�����ÿ�����ݵľ��룬ŷʽ���루�����Ͼ��룩
        distance=zeros(200,1);
        for i=1:200
            distance(i)=sqrt((test_data(1)-data(i,1)).^2+(test_data(2)-data(i,2)).^2);
        end
        %ѡ�����򷨣�ֻ�ҳ���С��ǰK������,�����ݺͱ�Ŷ���������
        for i=1:K
            ma=distance(i);
            for j=i+1:200
                if distance(j)<ma
                    ma=distance(j);
                    label_ma=label(j);
                    tmp=j;
                end
            end
            distance(tmp)=distance(i);  %������
            distance(i)=ma;
            label(tmp)=label(i);        %�ű�ǩ
            label(i)=label_ma;
        end
        cls1=0; %ͳ����1�о��������������ĸ���
        for i=1:K
            if label(i)==1
                cls1=cls1+1;
            end
        end
        cls2=K-cls1;    %��2�о��������������ĸ���
        if cls1>cls2
            plot(movenum,kissnum, 'k.'); %������1������Ƭ�������ݻ�С�ڵ�
        else
            plot(movenum,kissnum, 'g*'); %������2������Ƭ�������ݻ���ɫ*
        end
        label=[label1;label2]; %����label��ǩ����
    end
end
