---
layout: page
group: "deepimgir"
title: "Deep Image Retrieval"
link_url: https://arxiv.org/abs/1610.07940
---

- **목표** : 이미지를 질의로 받아 유사한 이미지를 검색하자.

- 논문
    - [Particular object retrieval with integral max-pooling of CNN activations](https://arxiv.org/pdf/1511.05879v1.pdf){: target="_blank"}
    - [Deep Image Retrieval: End-to-end learning of deep visual representations for image retrieval](http://www.europe.naverlabs.com/Research/Computer-Vision/Learning-Visual-Representations/Deep-Image-Retrieval){:target="_blank"}
    - [Beyond instance-level image retrieval: Leveraging captions to learn a global visual representation for semantic retrieval](http://openaccess.thecvf.com/content_cvpr_2017/papers/Gordo_Beyond_Instance-Level_Image_CVPR_2017_paper.pdf){:target="_blank"}

## 소개

- Instance-level image retrieval
    - 간단하게 생각하면 이미지를 넣어 유사한 이미지를 반환해주는 검색
    - 좀 더 구체적으로는 이미지 내 오브젝트와 가장 유사한 오브젝트를 찾아서 반환해주는 시스템.
    - 단순하게 동일한 이미지를 찾는 문제 수준부터 시멕틱한 정보까지 활용하여 유사한 이미지를 찾는 것까지 고려할 수 있다.
    - 보통 매우 큰 이미지 집합 내에서 검색을 해준다.
    - 웹 환경에서 사용되거나 사용자 앨범 등의 이미지에서 검색을 제공하는 등 여러 응용 예제가 있다.
- CNN 응용 예제들.
    - CNN을 이용하여 유사 이미지를 찾고자 하는 시도를 많이 했음.
        - ImageNet 으로 학습된 분류용 pretrained Network 을 얻어다가,
        - 분류 단계 이전 CNN Feature 정보로부터 유사한 이미지를 찾는 시도가 많았음.
        - 물론 성능은 대부분 적당하지만 부족한 것도 사실.
- 왜 부족할까?
    - 일단 데이터에 잡음이 많다. (유사 이미지를 찾기위해 사용되는 데이터로는 부족)
    - 사용된 모델이 부적절하다. (유사 이미지용 모델이 아님)
    - 잘못된 학습 방식 (기존의 분류 학습 방식으로는 유사 이미지를 찾지 못함)
- 다음은 흔하디 흔한 Transfer Leanring w/ CNN

![figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" height="350px"}

- 논문들에서는 검색 목적에 부합하는 Task로 정의하여 이 문제를 해결함.
    - 데이터를 최대한 유사 이미지 Task 에 맞도록 정제
    - 유사 이미지 검색을 위한 모델 개발
        - 검색에 유용한 정보를 추출하기 위해 **R-MAC** 과 같은 discriptor 를 사용
        - 이는 CNN 계열의 discriptor로 서로 다른 스케일의 이미지 Region을 추출하여 sum-aggregate 하는 방식
        - 물론 기존의 R-MAC 이론과는 조금 다르게 사용함.
    - 새로운 Loss 의 제안
        - Triplet 모델을 도입하여 유사 이미지 검색에 적합하도록 학습 구조를 변경
    - 추가로 유사한 이미지를 잘 찾을 수 있는 정보들을 추가
        - 이미지 서술 문장 등

- 먼저 각각의 모듈에 대한 기능을 살펴보고 전체 아키텍쳐를 조망해보도록 하자.

## R-MAC (Maximum Activations of Convolutions) Descriptor

- **R-MAC을** 이해하려면 먼저 **MAC** (Maximum Activations of Convolutions) Descriptor 가 무엇인지 알아야 한다.

- 논문  : [Particular object retrieval with integral max-pooling of CNN activations](https://arxiv.org/pdf/1511.05879v1.pdf){: target="_blank"}

![figure.2]({{ site.baseurl }}/images/{{ page.group }}/f02.png){:class="center-block" height="300px"}

- 먼저 pretrained CNN 을 사용하되 FC 레이어는 제거.
- Conv 레이어는 (\\(W \times H \times K\\)) 크기가 된다. 모두 Relu Activator 를 사용한다고 가정한다.
- \\(W \times H\\) 를 새로운 변수인 \\(X\\) 로 표현해보자. \\(X = \\{ X\_i \\}, i= \\{ 1, ..., K \\}\\)
    - 따라서 \\(X\_i\\) 는 2D Tensor를 나타내게 된다.
- 추가로 Conv의 position 정보를 \\(p\\) 하고 하자. 이 때 \\(X\_{i}(p)\\) 는 position \\(p\\) 에 해당하는 2차원 Tensor를 의미하게 된다.
- 새로운 feature vector \\(f\\) 를 정의한다.

$${\bf f}_{\Omega} = [ f_{\Omega, 1} ... f_{\Omega, i}...f_{\Omega, K}]^{T}, with\; f_{\Omega, i} = {\max}_{p \in \Omega} X_{i}(p)\qquad{(1)}$$

- 결국 `K` 크기를 가지는 1차원 벡터가 된다.
- 가장 중요한 단점은 모든 지역(localization) 정보가 사라진다.

- MAC 을 이용한 유사도 평가
    - 두 이미지로부터 얻은 MAC을 cos-sim 을 통해 유사도를 계산한다.
    - 보통은 맨 마지막 FCN을 사용.

- 실제 얻어진 결과를 보자.
    - 가장 활성화가 높은 위치의 정보만을 취득하여 유사도를 비교하기 때문에 거의 유사한 이미지에서는 동일한 지점에서의 Feature 값이 추출될 수 있다.
    - 아래 예제는 두 이미지 사이에서 추출된 MAC이 일치되는 영역 5개를 출력하고 있다.

![figure.3]({{ site.baseurl }}/images/{{ page.group }}/f03.png){:class="center-block" height="300px"}

### R-MAC

- Region feature vector
    - 앞서 이미지를 나타내는 feature vector \\(f\_{\Omega}\\) 를 살펴보았다.
    - 이것을 어떤 특정 사각형 Region에 해당하는 feature vector로 나타내면 다음과 같다.
    - \\(R \subseteq \Omega = \[1, W\] \times \[1, H\]\\)
    
$$f_{R} = [ f_{R, 1} ... f_{R, i} ... f_{R, K} ]^{T} \qquad{(2)}$$

- R-MAC
    - R-MAC은 그냥 크기가 다른 여러 개의 Region 을 사용한 것이라 생각하면 쉽다.
    - 일단 정사각형(square) 의 Region을 사용한다. (이미지의 \\(W\\) 와 \\(H\\) 가 다를 경우 짧은 쪽 기준)
    - 여러 스케일 값으로 Region 을 생성하게 되는데 이 때의 스케일을 결정하는 값을 Layer 라고 한다.
        - \\(l=1\\) 인 경우가 가장 큰 스케일을 가지게 된다.
        - 이 Region은 가용한 최대 면적의 40% 이하로는 떨어지지 않는 크기로 결정하게 된다.
    - 아래 그림을 보면 대충 어찌 뽑히는지 알 수 있다.
    
![figure.4]({{ site.baseurl }}/images/{{ page.group }}/f04.png){:class="center-block" height="400px"}

- 후처리로 각각의 Region별 MAC 을 합산한 뒤 Normalization 과정을 거치게 된다.
    - 여기서는 PCA 를 이용하여 Whitening 작업을 수행하게 된다.
    - 이후 DeepIR 에서는 이 부분을 따로 계산하는게 귀찮았는지 *shift + FC* 의 NN 모델로 대체하게 된다.

### Object Localization

- Region 내에서 Max 값을 구하는 것은 비싼 비용이다. Approximate integral max-pooling 을 이용하여 근사하자.

$$\tilde{f}_{R, i} = \left(\sum_{p \in R}X_{i}(p)^{\alpha}\right)^{\frac{1}{\alpha}} \simeq \max_{p \in R} X_{i}(p) = f_{R, i}\qquad{(3)}$$

- 제시된 Region 중 입력된 q 와 가장 유사한 Region 을 찾는 식은 다음과 같다.

$$\tilde{R} = {arg \max}_{ R \subseteq \Omega } \frac{ \tilde{f}_R^T {\bf q} }{ \| \tilde{f}_R \| \|{\bf q}\|}$$

- 실험 결과는 다음과 같다.

![figure.5]({{ site.baseurl }}/images/{{ page.group }}/f05.png){:class="center-block" height="150px"}

- 예제를 통한 결과는 다음과 같다.

![figure.6]({{ site.baseurl }}/images/{{ page.group }}/f06.png){:class="center-block" height="550px"}

## DeepIR

- 이전에 사용하던 R-MAC 을 여기에서는 RPN (Region Proposal Network)으로 변경
    - 그런데 기존의 R-MAC 도 테스트를 했음.
    - 게다가 사실 성능이 그리 차이가 나지 않는다.
    - 그래서 Resnet 버전에서는 다시 쉽고 편리한 R-MAC으로 변경
    
- 이 논문의 핵심은 Tripplet 임.
    - 샴(Siamese) 네트워크의 일종.
    - Query 이미지와 유사한 이미지 1개, 다른 이미지 1개를 이용하여 학습하는 구조.

![figure.7]({{ site.baseurl }}/images/{{ page.group }}/f07.png){:class="center-block" height="400px"}

$$L(I_q, I^{+}, I^{-}) = \max (0, m + q^Td^{-} - q^{T}d^{+})$$

- `m` 은 margin 값을 의미하고 `+` 는 positive sample, `-` 는 negative sample을 의미한다.
