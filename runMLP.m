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
pNumber_1  = input("\n\n 첫번째 층의 퍼셉트론(뉴런)의 개수를 입력해주세요. [1~inf] ");
pNumber_2  = input(" 두번째 층의 퍼셉트론(뉴런)의 개수를 입력해주세요. [1~inf] ");
eh  = input(" 학습을 몇번 반복할지 반복 횟수를 입력해주세요. [1~inf] ");
fprintf("\n\n %d-%d-%d-%d 의 구조를 갖는 신경망이 완성되었습니다.",imRe^2,pNumber_1,pNumber_2,10);
input(" 엔터를 누루면 학습을 시작합니다!");

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
        
        %% 그래프 보기
        tex2 = mean(P == O);
        tex1 = mean(E);
        mse(z,1) = mean(E);
        format shortG
        clc
        fprintf("    최종 학습 횟수 : %d번\n    총 학습된 글자 수 : %d 개 (한 번 반복에 64개씩 학습을 진행합니다.)\n    최종 손글씨 인식률 : %0.2f%%\n    전체 오차(MSE) : %0.5f",z,z*batch,round(tex2*100,4),round(tex1,4))
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
fprintf("    학습 횟수 : %d번\n    학습된 글자 수 : %d 개 (한 번 반복에 64개씩 학습을 진행합니다.)\n    손글씨 인식률 : %0.2f%%\n    전체 오차(MSE) : %0.5f",z,z*batch,round(tex2*100,4),round(tex1,4))
input("    학습이 완료되었습니다. 테스트 데이터로 실험을 해봅시다!")

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



