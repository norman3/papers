---
layout: page
group: "multigrain"
title: "MultiGrain: a unified image embedding for classes and instances."
link_url: https://arxiv.org/pdf/1902.05509.pdf
---

## Introduction

- 이미지 인식(Image recognition)은 컴퓨터 비젼의 핵심이며 매년 새로운 기법들이 쏟아져 나오고 있다.
- 하지만 대부분 인식 분야의 특정 영역에만 국한하여 문제를 해결한다.
- 큰 단위에서 작은 단위의 문제들을 살펴보면,
- <거시> to <미시> 방향으로 다음과 같은 예를 들어볼 수 있다. 
    - 어느 클래스에 속하는지를 찾는 문제 (intra-clas variation 에 상관 없이...)
    - 보여지는 상태에 상관없이 인스턴스 단위로 특정 객체가 무엇인지를 찾아내는 문제
    - 약간 수정된 이미지를 가지고 복사 여부를 확인하는 문제
- 특화된 영역에서는 좋은 성능을 모델을 만들 수 있지만 몇몇 분야에서는 이게 어려울 수 있다.
- 예를 들어 이미지 검색 (image retieval)등이 있다.
    - 최종 목표는 질의 이미지를 대규모 데이터베이스로부터 최적의 다른 이미지를 매칭하는 문제이다.
    - 어떤 경우 동일한 데이터베이스를 여러 입도(multi-granularites) 단위로 검색을 해야 할 수도 있다.
    - 왜냐하면 이미지 검색 시스템의 성능은 사용하는 이미지 임베딩(embedding)의 크기에 크게 영향받기 때문이다.
    - 결국 데이터베이스의 크기를 어떻게 가져갈 것인가에 대한 문제가 된다.
    - 동일 데이터베이스에 대해 여러 임베딩 데이터를 준비하게 되면 그만큼 리소스 비용 부담이 커지게 된다.
- 이 논문에서는 이미지에 대한 새로운 표현(representation) 방식을 소개한다.
- MultiGrain 이라 명명한 이 방식은 3가지 문제를 함께 처리한다.
  
![Figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" height="400" }

- 여기에서는 3가지 태스크를 한번에 학습한다.
    - 입도 수준이 다른 3가지 태스크를 한번에 학습하여 좋은 성능을 낸다. (개별적으로 학습한 것보다...)
- 인스턴스 단위의 이미지 검색은 다양한 형태로 실제 산업에서 사용된다.
    - (저작권이 있는) 복사된 이미지를 검출한다던지 처음 본 이미지를 인식한다던가 등
    - 대용량 이미지를 처리하는 경우 여러 문제를 다룰 수 있는 이미지 임베딩을 얻는 것이 중요하다.
    - 예를 들어 이미지 저장 플랫폼은 특정 분류를 수행하는 것 뿐만 아니라 동시에 복사본 여부 판별을 수행할 수도 있다.
    - 이러한 방법은 이미지를 저장하고 계산하는 실제 비용을 크게 감소시킬 수도 있다. (한번에 처리하니까)
- 이 관점에서 본다면 분류만을 위한 CNN 모델은 범용성은 쌈싸먹은 Feature 추출기라 할 수 있다.
- 분류 및 검색 둘다 잘 동작하는 임베딩을 학습할 수 있는 사실은 놀랍지만 서로 모순되는 것은 아니다.
    - 실제로 두 태스크 사이에는 논리적인 의존성이 있다.
    - 동일한 인스턴스를 포함하는 이미지는 정의에 따라 동일한 클래스에 속하게 된다.
    - 또한 복사된 이미지는 동일한 인스턴스를 가지고 있다.
- 일단 시작은 기존에 존재하는 분류용 CNN 네트워크를 활용한다. (자세한 내용은 뒤에서 보자)
- 분류와 인스턴스 인식은 cross-entropy 와 contrastive loss 를 사용한다.
    - 중요한 것은 인스턴스 인식 부분은 별도의 instance 레이블을 가진 학습 데이터를 요구하지 않는다는 것이다.

- 요약하자면,
    - MultiGrain 구조를 설명한다. 이 모델은 서로 다른 입도를 가지는 태스크에 잘 동작하는 임베딩을 제공한다.
    - 좋은 배치(batch) 전략을 사용하여 성능 이득을 얻는 것을 확인한다.
    - 이미지 검색 기법에서 영감을 받은 pooling layer 기법을 활용하여 정확도를 올린다.

## Related Work

- 간단하게 기술하고 넘어가자.
- image classification
    - ImageNet-2012 버전의 강자는 AmoebaNet-B (557M param., 480x480 input)
    - 이 논문에서는 ResNet-50 사용 (26M)
- image seaarch (from local feature to CNN)
    - 임베딩 유사도 문제로 문제를 해결한다.
    - 객체 검색을 위한 특별한 아키텍쳐
        - PCA-whitening
        - R-MAC image descriptor
- data augmentation
    - RA (repeated augmentation)

![Table.1]({{ site.baseurl }}/images/{{ page.group }}/t01.png){:class="center-block" height="400" }

## 아키텍쳐

- 일단 목표는 이미지 분류와 검색을 둘 잘 하는 모델을 만드는 것이다.

### Spatial pooling operators

- 여기서는 global spatial pooling layer 를 다룬다.
- 보통의 경우 pooling 을 사용하면 max pooling 을 사용하게 된다. 
- 반면 global spatial pooling 은 3d tensor 에 적용되는 pooling 기법이다.
- 분류
    - LeNet-5 나 AlexNet 에는 마지막 레이어에 spaital pooling 적용한다. 이는 위치 특성을 반영한다.
    - 하지만 ResNet, DenseNet 등은 average pooling 을 사용하고 있다.
- 이미지 검색
    - 이미지 검색의 경우 지역적 정보를 요구한다.
    - 특정 객체가 비주얼 적으로 유사한 경우 유사도가 높아야 한다.
    - 따라서 pooling 사용시 지역적(local) 특성을 담을 수 있는 pooling 을 사용해야 ㅎ나다.
- GeM (Generalized mean pooling)
    - 참고로 \\(p \\) 가 \\(\infty\\) 가 되면 max pooling
    - \\(p\\) 가 \\(1\\) 이 되면 avg. pooling 
    - 여기서 \\(p\\) 는 학습이 되는 weight.
    - 분류에서는 avg. pooling 을 사용하고 검색에서는 근사 max pooling 을 사용한다. (R-MAC 등)

$$ e = \left[\left(\frac{1}{|\Omega|}\sum_{u \in \Omega}{x^{p}_{cu}}\right)^{\frac{1}{p}}\right]_{c=1..C}\qquad{(1)}$$

- 이 논문에서는 처음으로 분류 문제에서도 GeM 을 활용한다.

![Figure.2]({{ site.baseurl }}/images/{{ page.group }}/f02.png){:class="center-block" height="500" }

### 목적 함수

- 검색과 분류를 결합하기 위해 각각의 목적 함수를 만들고 이를 결합하여 사용한다.

**분류용 Loss 함수**

- \\({\bf e}\_i \in R^d\\) 이미지 임베딩 벡터
- \\({\bf w}\_c \in R^d\\) 클래스 \\(c \in \{1,...,C\}\\) 를 위한 선형 분류기 파라미터
- \\(y\_i\\) 는 GT
- \\({\bf W} = [{\bf w}\_c]\_{c=1..C}\\)

$$L^{class}({\bf e}, {\bf W}, y_i) = -\langle{\bf w}_{y_i}, {\bf e}_i \rangle + \log{\sum_{c=1}^{C}{\exp\langle{\bf w}_c,{\bf e}_i\rangle}}\qquad{(2)}$$

**검색용 Loss 함수**

- 이미지 검색에서는 두개의 매칭 이미지를 사용한다. (positive pair)
- 이 경우 임베딩 값이 매칭되지 않은 이미지들보다 거리상으로 더 가까워야 한다.
- contrastive loss 는 positive pair 간의 거리를 최소 threshold 보다 더 가깝게 놓아두는 방법을 사용한다.
- 반면 triple loss 는 positive pair 를 negative pair 보다 더 가깝도록 구성한다.
- triple loss 의 경우 데이터를 구성하기가 매우 어렵다.
- 배치(batch) 단위의 이미지 처리를 하기 때문에 실제 구현이 까다롭다.
    - Wu 는 이런 어려움을 극복하고자 새로운 방법을 제안했다.
    - 주어진 배치 이미지들로부터 각 이미지의 임베딩을 단위 백터로 normalize 한 다음 margin loss 를 이용하여 거리 계산을 수행한다.
    - 이미지를 단위 구 (unit sphere) 공간에 사상하여 margin loss 를 만들어낸다.


$$Loss^{retr}({\bf e}_i, {\bf e}_j, \beta, y_{ij}) = \max\left\{0, \alpha+y_{ij}(D({\bf e}_i,{\bf e}_j)-\beta)\right\}\qquad{(3)}$$

- 여기서 \\(D\\) 는 Euclidean 거리이다. (물론 이미 normalize 되어 있다.)
    - \\(D({\bf e}\_i,{\bf e}\_j)=\|\|\frac{e\_i}{\|\|e\_i\|\|} - \frac{e\_j}{\|\|e\_j\|\|} \|\|\\)

