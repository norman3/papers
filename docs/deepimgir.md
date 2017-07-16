---
layout: page
group: "deepimgir"
title: "Deep Image Retrieval"
link_url: https://arxiv.org/abs/1610.07940
---

- **목표** : 이미지를 질의로 받아 유사한 이미지를 검색하자.

- 논문 : [Deep Image Retrieval](http://www.europe.naverlabs.com/Research/Computer-Vision/Learning-Visual-Representations/Deep-Image-Retrieval){:target="_blank"}
    - 'XRCE Research Europe' 에서 'Naver Labs Europe' 이 된 회사에서 나온 논문

## 소개

- Instance-level image retrieval
    - 간단하게 생각하면 이미지를 넣어 유사한 이미지를 반환해주는 검색
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

- 논문에서는 검색 목적에 부합하는 Task로 정의하여 이 문제를 해결함.
    - 데이터를 최대한 유사 이미지 Task 에 맞도록 정제
    - 유사 이미지 검색을 위한 모델 개발
        - 검색에 유용한 정보를 추출하기 위해 **R-MAC** 과 같은 discriptor 를 사용
        - 이는 CNN 계열의 discriptor로 서로 다른 스케일의 이미지 Region을 추출하여 sum-aggregate 하는 방식
        - 물론 기존의 R-MAC 이론과는 조금 다르게 사용함.
    - 새로운 Loss 의 제안
        - Triplet 모델을 도입하여 유사 이미지 검색에 적합하도록 학습 구조를 변경

![figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" height="400px"}

- 먼저 각각의 모듈에 대한 기능을 살펴보고 전체 아키텍쳐를 조망해보도록 하자.


## R-MAC (Maximum Activations of Convolutions) Descriptor

- 논문  : [Particular object retrieval with integral max-pooling of CNN activations](https://arxiv.org/pdf/1511.05879v1.pdf){: target="_blank"}

<!-- - 3 가지 주요 개선점 -->
<!--     - 단 한번의 입력으로 여러 지역 정보를 추출함 -->
<!--     - Max-Pooling 을 이용해 일반화된 평균값을 만들어 한개의 feature map을 구성함. -->
<!--     - 쿼리 확장 기능 -->

![figure.2]({{ site.baseurl }}/images/{{ page.group }}/f02.png){:class="center-block" height="300px"}

- 먼저 pretrained CNN 을 사용하되 FC 레이어는 제거.
- Conv 레이어는 (\\(W \times H \times K\\)) 크기가 된다. 모두 Relu Activator 를 사용한다고 가정한다.
- \\(W \times H\\) 를 다른 변수인 \\(X\\) 로 표현해보자. \\(X = \\{ X\_i \\}, i= \\{ 1, ..., K \\}\\)
    - 따라서 \\(X\_i\\) 는 2D Tensor를 나타내게 된다.
- Conv의 position 정보를 \\(p\\) 하고 하자. \\(X\_{i}(p)\\) 는 position \\(p\\) 에 해당하는 Tensor를 의미하게 된다.

- 새로운 feature vector \\(f\\) 를 정의한다.

$${\bf f}_{\Omega} = [ f_{\Omega, 1} ... f_{\Omega, i}...f_{\Omega, K}]^{T}, with\; f_{\Omega, i} = {\max}_{p \in \Omega} X_{i}(p)\qquad{(1)}$$
