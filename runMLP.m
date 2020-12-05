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
pNum_1  = input("\n\n ù��° ���� �ۼ�Ʈ��(����)�� ������ �Է����ּ���. [1~inf] ");
pNum_2  = input(" �ι�° ���� �ۼ�Ʈ��(����)�� ������ �Է����ּ���. [1~inf] ");
eh  = input(" �н��� ��� �ݺ����� �ݺ� Ƚ���� �Է����ּ���. [1~inf] ");
fprintf("\n\n %d-%d-%d-%d �� ������ ���� �Ű���� �ϼ��Ǿ����ϴ�.",imRe^2,pNum_1,pNum_2,10);
input(" ���͸� ����� �н��� �����մϴ�!");

% 1st Layer
node_1w = fc_node('weight', imRe^2, pNum_1);
node_1b = fc_node('bias', pNum_1,1);
% 2nd Layer
node_2w = fc_node('weight', pNum_1, pNum_2);
node_2b = fc_node('bias', pNum_2,1);
% 3rd Layer
node_3w = fc_node('weight', pNum_2, 10);
node_3b = fc_node('bias', 10,1);
% batch size
batch =64;

close all
for z = 1 : eh
    
% data shuffle
p = randperm(cols);                                           
X = x(:,p(1:batch));
Y = y(p(1:batch),:);

% batch memory init (weight)
batch_1 = 0; batch_2 = 0; batch_3 = 0; 
% batch memory init (bias)
batch_4 = 0; batch_5 = 0; batch_6 = 0;
        
    for i = 1 : batch    

%% Feed Forward propagation

f1 = relu(X(:,i)' * node_1w + node_1b');
f2 = relu(f1 * node_2w + node_2b');
f3 = exp(f2 * node_3w + node_3b') / sum(exp(f2 * node_3w + node_3b')) ;
        
%% Error
P(i) = find(f3==max(f3));
O(i) = find(Y(i,:)==max(Y(i,:)));
E(i,:) = - sum(Y(i,:).*log(f3));
        
        
%% Back propagation  

b3 = f3 - Y(i,:);    
b2 = b3 * node_3w' .* reluGradient(f2);     
b1 = b2 * node_2w' .* reluGradient(f1);
       
        %% Batch 
         batch_1 = batch_1 + (alpha * f2' * b3);        
         batch_4 = batch_4 + (alpha * b3)'; 
         
         batch_2 = batch_2 + (alpha * f1' * b2);   
         batch_5 = batch_5 + (alpha * b2)';
    
         batch_3=  batch_3 + (alpha * X(:,i) * b1);
         batch_6 = batch_6 + (alpha * b1)' ;
         
    end
    
        %% Update
        node_3w = node_3w - batch_1 / batch;
        node_2w = node_2w - batch_2 / batch;
        node_1w = node_1w - batch_3 / batch;
        
        node_3b = node_3b - batch_4 / batch;
        node_2b = node_2b - batch_5 / batch;
        node_1b = node_1b - batch_6 / batch;
        
        %% �׷��� ����
        tex2 = mean(P == O);
        tex1 = mean(E);
        mse(z,1) = mean(E);
        format shortG
        clc        
        fprintf("�н� Ƚ�� : %d��\n",z)
        fprintf("�н��� ���� �� : %d �� (�� �� �ݺ��� 64���� �н��� �����մϴ�.)\n",z*batch)
        fprintf("�н� ������ �ձ۾� �νķ� : %0.2f%%\n",round(tex2*100,4))
        fprintf("��ü �н� ����(MSE) : %0.5f",round(tex1,4))
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
fprintf("�н� Ƚ�� : %d��\n",z)
fprintf("�н��� ���� �� : %d �� (�� �� �ݺ��� 64���� �н��� �����մϴ�.)\n",z*batch)
fprintf("�н� ������ �ձ۾� �νķ� : %0.2f%%\n",round(tex2*100,4))
fprintf("��ü �н� ����(MSE) : %0.5f",round(tex1,4))
input("    �н��� �Ϸ�Ǿ����ϴ�. �׽�Ʈ �����ͷ� ������ �غ��ô�!")

%%
load test\test_input.mat; 
load test\test_output.mat; 

results = [];
for i= 1 : 10000
    f1 = relu(test(:,i)' * node_1w + node_1b');
    f2 = relu(f1 * node_2w + node_2b');
    f3 = exp(f2 * node_3w + node_3b') / sum(exp(f2 * node_3w + node_3b'));
    results(i) =  min( yy(i,:) == (f3 ==max(f3)));
end

fprintf("��ü �׽�Ʈ ������ �н� ��� %0.2f %% ��Ȯ��\n",round(mean(results)*100,2))

results = [];
for i= 1 : 10
    f1 = relu(test(:,i)' * node_1w + node_1b');
    f2 = relu(f1 * node_2w + node_2b');
    f3 = exp(f2 * node_3w + node_3b') / sum(exp(f2 * node_3w + node_3b'));
    results(i) =  min( yy(i,:) == (f3 ==max(f3)));
    
    im = reshape(test(:,i)',28,28);
    imshow(im);
    title(find(f3 ==max(f3))-1)
    input("")
end


