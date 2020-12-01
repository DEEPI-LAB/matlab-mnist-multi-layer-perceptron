% *********************************************
% MNIST Neural Networks
% @author: Deep.I Inc. @Jongwon Kim
% deepi.contact.us@gmail.com
% Revision date: 2020-12-01
% See here for more information :
%    https://deep-eye.tistory.com
%    https://deep-i.net
% **********************************************
% STRUCTURE : BATCH MLP
% input : 787 x 60000
% output : 10 x 60000
% MODE : Batch
% ACTIVATION FUNCTION : 'Relu'
% ERROR RATE : 2.51

%% F5�� ������ �������ּ���.
clear all
clc
cla
close all
input("\n\n �ۼ�Ʈ���� Ȱ���� �ձ۾� �ν� ���α׷� �Դϴ�. [����Ű�� �����ּ���] ")

%%
load train\train_input.mat; 
load train\train_output.mat; 

clc
input("\n\n �����ص帰 ���� �����͸� �ε��ҰԿ�. [����Ű�� �����ּ���] ")                                              
                                              
%% �н� �����͸� �ѹ� ���ô�                                              
mnist = images(:,1:200);                        
mnist = reshape(mnist,28,28,200); 
montage(mnist)                                      

title("�н��� Ȱ���� �ձ۾� �̹��� �Դϴ�. [����Ű�� �����ּ���] ")
clc
input("\n\n �н��� Ȱ���� �ձ۾� �̹��� �Դϴ�. [����Ű�� �����ּ���] ")

x = images;
cols = size(x,2);
imRe = 28;

clc
fprintf("�������� ���� : %d \n�̹��� �ػ� : %d x %d\n�Է� ���� : %d\n",cols,imRe,imRe,imRe^2)
alpha = input("\n\n �н��� ������ �����ϴ� Learning Rate�� �Է����ּ���. [0.1~ 0.0001] ");
clc
pNumber_1  = input("\n\n ù��° ���� �ۼ�Ʈ��(����)�� ������ �Է����ּ���. [1~inf] ");
pNumber_2  = input(" �ι�° ���� �ۼ�Ʈ��(����)�� ������ �Է����ּ���. [1~inf] ");
eh  = input(" �н��� ��� �ݺ����� �ݺ� Ƚ���� �Է����ּ���. [1~inf] ");
fprintf("\n\n %d-%d-%d-%d �� ������ ���� �Ű���� �ϼ��Ǿ����ϴ�.",imRe^2,pNumber_1,pNumber_2,10);
input(" ���͸� ����� �н��� �����մϴ�!");

fc_w1 =0.1*randn(imRe^2,pNumber_1);
fc_b1  = 0.1*randn(pNumber_1,1);
fc_w2 =0.1*randn(pNumber_1,pNumber_2);
fc_b2  =0.1*randn(pNumber_2,1);
fc_w3 =0.1*randn(pNumber_2,10);
fc_b3  = 0.1*randn(10,1);

batch =64;
close all
for z = 1 : eh
    
        p = randperm(cols);                                           
        X = x(:,p(1:batch));
        Y = y(p(1:batch),:);
        
        batch_1 = 0; batch_2 = 0; batch_3 = 0;
        batch_4 = 0; batch_5 = 0; batch_6 = 0;
        
       
    for i = 1 : batch    

        %% Feed Forward propagation
        
        Z6 = relu(X(:,i)'*fc_w1 + fc_b1');
        Z7 = relu(Z6*fc_w2 + fc_b2');
        Z8 =exp(Z7*fc_w3 + fc_b3') / sum(exp(Z7*fc_w3 + fc_b3')) ;
        
        P(i) = find(Z8==max(Z8));
        O(i) = find(Y(i,:)==max(Y(i,:)));
        E(i,:) =  - sum(Y(i,:).*log(Z8));
        
        %% Back propagation  
        
        D8 = Z8 - Y(i,:);    
        D7 = D8*fc_w3'.*reluGradient(Z7);     
        D6 = D7*fc_w2'.*reluGradient(Z6);
       
         batch_1 = batch_1 + (alpha*Z7'*D8);        
         batch_4 = batch_4 + (alpha*D8)'; 
         
         batch_2 = batch_2 + (alpha*Z6'*D7);   
         batch_5 = batch_5 + (alpha*D7)';
    
         batch_3=  batch_3 + (alpha*X(:,i)*D6);
         batch_6 = batch_6 + (alpha*D6)' ;
         
    end
        fc_w3 = fc_w3 - batch_1/ batch;
        fc_w2 = fc_w2 - batch_2/ batch;
        fc_w1 = fc_w1 - batch_3/ batch;
        
        fc_b3 = fc_b3 - batch_4/ batch;
        fc_b2 = fc_b2 - batch_5/ batch;
        fc_b1 = fc_b1 - batch_6/ batch;
        
        %% �׷��� ����
        tex2 = mean(P == O);
        tex1 = mean(E);
        mse(z,1) = mean(E);
        format shortG
        clc
        fprintf("    ���� �н� Ƚ�� : %d��\n    �� �н��� ���� �� : %d �� (�� �� �ݺ��� 64���� �н��� �����մϴ�.)\n    ���� �ձ۾� �νķ� : %0.2f%%\n    ��ü ����(MSE) : %0.5f",z,z*batch,round(tex2*100,4),round(tex1,4))
        cla
        subplot(1,2,1)
        plot(mse);
        axis([0 inf 0 5])
        title("MSE")
        drawnow;
        subplot(1,2,2)
        
        testing = reshape(X,28,28,64);
        montage(testing(:,:,1:20));
        title("�н����� ����")
        drawnow;
        %
end
clc
fprintf("    �н� Ƚ�� : %d��\n    �н��� ���� �� : %d �� (�� �� �ݺ��� 64���� �н��� �����մϴ�.)\n    �ձ۾� �νķ� : %0.2f%%\n    ��ü ����(MSE) : %0.5f",z,z*batch,round(tex2*100,4),round(tex1,4))
input("    �н��� �Ϸ�Ǿ����ϴ�. �׽�Ʈ �����ͷ� ������ �غ��ô�!")

T = imread('sample\sample_1.png'); 

TEST =logical(T(:,:,1));
TEST = double(TEST(:));
Z6 = relu(TEST'*fc_w1 + fc_b1');
Z7 = relu(Z6*fc_w2 + fc_b2');
Z8 =exp(Z7*fc_w3 + fc_b3') / sum(exp(Z7*fc_w3 + fc_b3')) ;

RESULT = find(Z8==max(Z8));

close all
imshow(imresize(T,10))
title(RESULT)



