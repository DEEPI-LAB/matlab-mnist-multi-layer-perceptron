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

%% F5를 눌러서 실행해주세요.

clear all
clc
cla
close all
input("\n\n 퍼셉트론을 활용한 손글씨 인식 프로그램 입니다. [엔터키를 눌러주세요] ")

%%
load train\train_input.mat; 
load train\train_output.mat; 

clc
input("\n\n 배포해드린 숫자 데이터를 로드할게요. [엔터키를 눌러주세요] ")                                              
                                              
%% 학습 데이터를 한번 봅시다                                              
mnist = images(:,1:200);                        
mnist = reshape(mnist,28,28,200); 
montage(mnist)                                      

title("학습에 활용할 손글씨 이미지 입니다. [엔터키를 눌러주세요] ")
clc
input("\n\n 학습에 활용할 손글씨 이미지 입니다. [엔터키를 눌러주세요] ")

x = images;
cols = size(x,2);
imRe = 28;
    
clc
fprintf("데이터의 개수 : %d \n이미지 해상도 : %d x %d\n입력 차원 : %d\n",cols,imRe,imRe,imRe^2)
alpha = input("\n\n 학습의 정도를 결정하는 Learning Rate을 입력해주세요. [0.1~ 0.0001] ");
clc
pNum_1  = input("\n\n 첫번째 층의 퍼셉트론(뉴런)의 개수를 입력해주세요. [1~inf] ");
pNum_2  = input(" 두번째 층의 퍼셉트론(뉴런)의 개수를 입력해주세요. [1~inf] ");
eh  = input(" 학습을 몇번 반복할지 반복 횟수를 입력해주세요. [1~inf] ");
fprintf("\n\n %d-%d-%d-%d 의 구조를 갖는 신경망이 완성되었습니다.",imRe^2,pNum_1,pNum_2,10);
input(" 엔터를 누루면 학습을 시작합니다!");

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
        
        %% 그래프 보기
        tex2 = mean(P == O);
        tex1 = mean(E);
        mse(z,1) = mean(E);
        format shortG
        clc        
        fprintf("학습 횟수 : %d번\n",z)
        fprintf("학습된 글자 수 : %d 개 (한 번 반복에 64개씩 학습을 진행합니다.)\n",z*batch)
        fprintf("학습 데이터 손글씨 인식률 : %0.2f%%\n",round(tex2*100,4))
        fprintf("전체 학습 오차(MSE) : %0.5f",round(tex1,4))
        cla
        subplot(1,2,1)
        plot(mse);
        axis([0 inf 0 5])
        title("MSE")
        drawnow;
        subplot(1,2,2)
        
        testing = reshape(X,28,28,64);
        montage(testing(:,:,1:20));
        title("학습중인 숫자")
        drawnow;
        %
end
clc
fprintf("학습 횟수 : %d번\n",z)
fprintf("학습된 글자 수 : %d 개 (한 번 반복에 64개씩 학습을 진행합니다.)\n",z*batch)
fprintf("학습 데이터 손글씨 인식률 : %0.2f%%\n",round(tex2*100,4))
fprintf("전체 학습 오차(MSE) : %0.5f",round(tex1,4))
input("    학습이 완료되었습니다. 테스트 데이터로 실험을 해봅시다!")

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

fprintf("전체 테스트 데이터 학습 결과 %0.2f %% 정확도\n",round(mean(results)*100,2))

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


