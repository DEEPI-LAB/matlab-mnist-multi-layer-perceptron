# MNIST DATASET

**MNIST 데이터셋**은 머신러닝을 입문하는 분들이 처음 접하게 되는 데이터 중 하나입니다. 28 x 28 해상도를 가지는 흑백 이미지로 구성되어있지만, 영상 처리 알고리즘 이외 **K-Measn, PCA, RNN** 등 다양항 기법이 적용 가능하여 초기 데이터 분석 단계에서 연습에 활용되고 있습니다.
<br/>
![image_1](https://blog.kakaocdn.net/dn/ca7ret/btqPaVQUEYp/KksWbG9M8uVDyCCA4GJ0w1/img.png)
<br/>
저 역시, 처음 머신러닝을 시작했을때 XOR 논리 게이트 문제와 함께 자주 사용했었던 데이터입니다. 저 작은 크기의 데이터 하나 가지고도 머리 아파했던 기억이납니다. 만약 처음 입문하시는 분이라면 이 데이터 하나만 가지고 다양한 시도와 분석을 먼저 해보시길 바랍니다. 정말 많은 도움이 됩니다.

# **MULTI-LAYER PERCEPTRON**
![image2](https://blog.kakaocdn.net/dn/b4Zwjs/btqPgHDNy1V/3y8ok8bAWyHxvsUgYZpk10/img.png)
<br/>
직접적인 이론은 생략하도록 하겠습니다. 2개의 은닉층을 가지는 MLP 구조로 설계하도록 하겠습니다. MLP는 CNN처럼 2차원 영상이 입력될 수없기때문에 1차원으로 데이터를 변환시켜줘야 합니다. **28 X 28** 이미지는 **784 개의 1차원 벡**터로 변환되어 N개의 노드를 가지는 은닉층에 입력됩니다. 첨부된 데이터는 **총 60,000개이며 784개의 차원을 가지므로 60,000 x 784 의 형태를 가지게 됩니다.**

# **Algorithm**
**0. 샘플코드 다운로드**
**[github.com/DEEPI-LAB/matlab-mnist-multi-layer-perceptron.git](https://github.com/DEEPI-LAB/matlab-mnist-multi-layer-perceptron.git)**

    git clone https://github.com/DEEPI-LAB/matlab-mnist-multi-layer-perceptron.git
  또는 위 링크에서 좌측 상단 초록색 CODE를 클릭하시면 하단에 Download ZIP을 클릭하시면 됩니다.
  <br/>
  ![image3](https://blog.kakaocdn.net/dn/xlMuK/btqO71qzhbM/btOFw0JZj7v9iiYz2VpKp1/img.png)
<br/>
학생들에게 배포해준 자료이다 보니, 아기자기한 표현이 많은 코드입니다. CNN 알고리즘과 비교하여 단순한 기법이지만 MLP 기반 MNIST 학습 성능은  **평균적으로 95 ~ 98% 수준**으로 높습니다. 첨부된 코드에서 학습 성능 향상을 위해 개선된 부분은 다음과 같습니다.

  

-   **데이터 셔플링, ReLu 함수, 크로스 앤트로피 오차**

  

현대의 신경망에는 이외에 다양한 기술적 테크닉이 존재합니다. 함수 구현이 아닌, 이론을 통해 이해가 된 알고리즘을 자신만의 코드로 변환하여 MNIST의 성능을 향상시켜보시길 바랍니다.