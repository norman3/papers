---
layout: page
group: "do_better_imagenet_models_transfer_better"
title: "Do Better ImageNet Models Transfer Better?"
link_url: https://arxiv.org/abs/1805.08974
---

## 초록

- Transfer Learning 은 computer vision 에서 매우 성공적인 모델.
- 여기에는 슬픈 전설이 있다.
    - ImageNet 에 대해 더 성능이 우수한 모델을 backbone 으로 하여 Transfer learning을 하면 성능이 더 좋다는 것.
    - 이 논문에서는 12개의 데이터 집합, 16개의 분류 모델을 비교하여 이 전설이 사실인지를 검증해본다.
    - 실제 backbone 과 transfer task 와의 성능에 대한 상관 관계가 매우 높음을 확인했다.

## 소개

- 지난 십여년간 computer vision 학계에서는 벤츠마크를 위한 측정 수단을 만들어내는데 공을 들였다. (~~본격 대결을 위한 기준 자료~~)
- 그 중 가장 성공한 프로젝트는 ImageNet 이다.
- 이 데이터를 기반으로 전이 학습(transfer learning), 객체 탐지(object detection), 세그멘테이션(image segmentation) 등의 다양한 분야에 대한 평가를 수행했다.
- 여기에서의 암묵적인 가정은,
    - 첫번째는 ImageNet 에서 좋은 성능을 보이는 모델은 다른 Image Task 에서도 좋은 성능을 낸다는 것.
    - 두번재는 더 좋은 모델을 사용하면 전이(transfer) 학습에서 더 좋은 성능을 얻을 수 있다는 것.
- 이전의 다양한 연구들을 토대로 보자면 이 가정은 어느 정도 들어맞는 것 같다.
- 이 논문에서는 실험 기준을 세우기 위해 ImageNet feature 와 분류 모델을 모두 살펴본다.
    - 16개의 최신 CNN 모델들과 12개의 유명한 분류용 데이터셋을 사용하여 검증한다.
- 총 3가지 실험을 수행하였다.
    - 1. 기 학습된 ImageNet에서 고정된 feature 값을 추출한 뒤 이 결과로 task 를 학습
    - 2. 기 학습된 ImageNet을 다시 fine-turning 하여 학습
    - 3. 그냥 모델들을 개별 task 에 스크래치 테스트를 수행
- 결론을 요약해보자면 다음과 같다.
    - ImageNet 모델을 사용하는 것이 feature 만 활용하는 것보다 성능이 좋다.
        - 선형 분류 task 대상 (r=0.99)
    - 가장 좋은 것은 fine-tuning 을 하는 모델이다. (r=0.96)
    - 정칙화(Regularization)를 적용하여 성능을 끌어올린 ImageNet 모델에서 고정된 feature 추출하는 방식(1)의 성능이 매우 안 좋다.
    - ImageNet 에서 좋은 성능을 보이는 모델은 어떤 task 이던 비슷하게 성능이 좋다.
        - FGVC task 2개를 실험한 결과 성능이 매우 좋음을 확인했다.
    

## Statistical methods

- 서로 다른 난이도를 가진 여러 데이터 집합을 통해 각 모델 성능의 상관 관계를 제대로 측정하는 것은 매우 어려운 일이다.
- 그래서 단순하게 성능이 몇 % 올랐는지를 확인하는 방식에는 문제가 있다.
    - "성능이 50%인 상태 vs. 성능이 99% 상태" 에서 1% 의 성능 향상은 서로 다른 의미를 가진다.
- 여기서는 log-odd 를 사용하여 변환된 성능 측정 방식을 사용하였다.

$$logit(p) = \log{\frac{p}{(1-p)}} = sigmoid^{-1}(p)$$

- logit 변환은 비율 데이터 분석에서 가장 흔하게 사용도는 계산 방식이다.
- 사용되는 스케일이 \\(\log\\) 단위로 변경되었기 때문에 값의 변화량 \\(\Delta\\) 는 \\(\exp\\) 비율로 적용되는 것을 알 수 있다.

$$logit{\left(\frac{n_{correct}}{n_{correct}+n_{incorrect}}\right)} + \Delta = \log{\left(\frac{n_{corrent}}{n_{incorrect}}\right)} + \Delta = \log{\left(\frac{n_{correct}}{n_{incorrect}}\exp{\Delta}\right)}$$

- Error bar 도 Morey 가 제안한 방법으로 뭐 적당히 잘 구성함. (논문 참고)
- 이제 ImageNet 정확도와 log-transformed 정확도의 상관 관계를 측정한다.
    - 이에 대한 자세한 내용은 Appendix 를 참고하면 된다.

![Figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" height="400" }

## 결과

- 16개의 모델로 ImageNet(ILSVRC-2012) validation의 top-1 정확도를 비교 (각 모델들은 71.6%~80.8% 성능)
- 각 모델들은 크게보면 Inception, ResNet, DenseNet, MobileNet, NASNet 으로 구분지을 수 있다.
- 공평한 비교를 위해 모든 모델은 직접 재학습하였다.
    - 이 때 BN-scale-parameter, Label-smoothin, dropout auxiliary-head 등을 나누어 확인했다.
    - Appendix A.3 에 좀 더 자세히 기술하였다.
- 총 12개의 분류용 데이터를 실험하였다.
    - 데이터의 크기는 2,040개에서부터 75,750 등 다양하다.
    - CIFAR-10, CIFAR-100, PASCAL-VOC-2007, Caltech-101, Food-101, Bird-snap, Cars, Aircraft, Pets, DTD, scene-classification etc.


![Table.1]({{ site.baseurl }}/images/{{ page.group }}/t01.png){:class="center-block" height="350" }

- 그림.2 가 실제 테슽트 결과를 나타낸 그림이다.
- ImageNet 에 대한 top-1 결과와 새로운 task 에 대한 상관 관계를 나타내고 있다.
- 다음 설정으로 실험하였다.
    - (1) logistic regression classifier. (마지막 전 레이어 fixed feature 사용.)
    - (2) ImageNet 을 기본으로 fine-tuning 작업을 수행
    - (3) 동일한 아키텍쳐 모델로 새로운 task 에서 재학습

![Figure.2]({{ site.baseurl }}/images/{{ page.group }}/f02.png){:class="center-block"}

### fixed features

- fixed feature 를 먼저 추출한 뒤 이 값으로 logistic regression 을 수행
    - L-BFGS 를 사용하였고 data augmentation은 비교를 위해 사용 안함
- 최대한 실험 조건을 서로 맞추어 실험하였다.
- 공개적으로 오픈된 체크 포인트를 활용하여 테스트를 한 경우,
    - ResNet 과 DenseNet 이 다른 모델에 비해 일관적으로 높은 성능을 얻음을 확인하였다.
    - 다만 ImageNet 과 transfer 정확도 사이의 상관 관계가 매우 낮았다. (부록 B 참고)
    - 이는 정규화 적용 유무에 대한 차이로 보여진다.

- 그림.3 은 정규화 방식을 다르게 적용하여 성능을 확인한 것이다.
- 총 4개의 종류로 여러 개의 방식을 조합해서 성능을 확인해본다.
    - (1) BN scale parameter 제거 (\\(\gamma\\) 파라미터)
    - (2) label smoothing 적용
    - (3) dropout
    - (4) auxiliary classifiert head 사용
- 이들은 ImageNet top-1 정확도에 1% 이내의 성능 영향을 미친다.
- 하지만 transfer 정확도에는 각각이 미치는 영향도가 다름을 확인하였다.

![Figure.3]({{ site.baseurl }}/images/{{ page.group }}/f03.png){:class="center-block" height="500"}

- Embedding 에서의 차이도 있음.

![Figure.4]({{ site.baseurl }}/images/{{ page.group }}/f04.png){:class="center-block" height="300"}

- 이것과 관련해서는 설명이 좀 이상하다. (~~뭔 말인지 모르겠다.~~)(부록.C.1 를 살펴보아야 할 듯)


### fine-tuning

- 그림.2 가 fine-turning 결과이다.
- ImageNet 으로 학습된 결과에 fine-tuning 을 각 task 에 대해 수행한다.
- 20,000 step 의 학습을 수행하고 Nesterov momentum 과 cosine-decay lr 을 적용하였다.
- batch_size 는 256 이다.
- grid search 기법을 이용하여 최적의 lr 과 weight decay 를 선정하였다. (부록. A.5)
- 이 실험에서 ImageNet top-1 정확도와 transfer 정확도의 상관관계가 높음을 확인하였다. (r=0.96)
- logistic regression 과 비교해서 정칙화(regularization) 기법과 학습시 사용한 설정값들은 영향도가 작다는 것을 확인했다.
    - 그림 5는 Inception-v4 와 Inception-ResNet-v2 에 대해 마찬가지로 정칙화를 적용해본 결과이다.
    
![Figure.5]({{ site.baseurl }}/images/{{ page.group }}/f05.png){:class="center-block" height="400"}

- logistic regression 실험과 마찬가지로 BN scale 은 사용하고 label smoothing 은 사용하지 않는 것이 좋다.
- dropout 과 auxiliary head 는 경우에 따라 성능 향상이 되기도 한다.
- 이것은 부록.C2 에 좀 더 설명하였다.

- 무엇보다도 중요한 것은 fine-tuning 모델이 transfer 정확도에 가장 좋은 성능을 낸다는 것이다.
    - 하지만 사용하는 데이터셋의 종류에 따라 얻을 수 있는 이득의 정도는 다 다르다.
- 그림.6 과 부록.E 를 살펴보자.

![Figure.6]({{ site.baseurl }}/images/{{ page.group }}/f06.png){:class="center-block"}


### Random initialization

- 앞선 결과만 살펴본다면 성능의 향상이 ImageNet 에서 학습된 weight 정보로부터 오는 것인지, 아니면 아키텍쳐 자체에서 오는 것인지 확인하기가 어렵다.
- 그래서 여기서는 순수하게 동일한 아키텍쳐만 사용하는 방법으로 학습을 진행한다.
    - 사용된 옵션들은 fine-tuning 한 것과 동일하게 사용한다.
- 이 때의 상관 관계는 어느 정도 유의미한 것으로 확인되었다. (r=0.55)
- 특히 10,000개 미만의 데이터를 가지는 7개의 데이터집합에서는 상관 관계가 매우 낮았다. (r=0.29)
    - 부록.D 를 살펴보자.
- 반면 큰 데이터 크기를 가지는 데이터집합에서는 상관 관계가 매우 높았다. (r=0.86)

### Fine-tuning with better models is comparable to specialized methods for transfer learning

- ImageNet 정확도와 transfer 정확도가 높은 상관 관계에 놓여있다고 확있되었으므로...
- ImageNet 에서 더 좋은 모델이 transfer 에도 더 좋은지 확인해보자.
- 그림.6 에서 보면 12개의 데이터 집합 중 7개의 데이터집합에서 SOTA 를 찍었다. (부록.F)
    - 이 말은 ImageNet 으로 학습한 모델의 성능이 transfer 성능에 큰 영향을 준다는 의미이다.


### ImageNet pretraining does not necessarily improve accuracy on fine-grained tasks

![Figure.7]({{ site.baseurl }}/images/{{ page.group }}/f07.png){:class="center-block"}

- FGVC Task 에서는 transfer 에도 성능 향상 폭이 적다.
    - Stanford Car, FGVC-Aircraft 등
    - ImageNet 에서 car 클래스는 10개 정도이다. 그래서 Stanford Car 는 잘 안되는 듯. (10 vs. 196)

###  ImageNet pretraining accelerates convergence

![Figure.8]({{ site.baseurl }}/images/{{ page.group }}/f08.png){:class="center-block"}

- Stanford Car 와 FGVC-Aircraft 에서 [2]와 [3] 의 방식에 따른 성능 차이가 없는 것을 확인하였다.
- 그러면 학습 속도에서의 차이도 있을까?
- 위 그림을 보면 [2] 의 방식이 수렴 속도에서 차이가 있는 것을 확인할 수 있다.

### Accuracy benefits of ImageNet pretraining fade quickly with dataset size

![Figure.9]({{ site.baseurl }}/images/{{ page.group }}/f09.png){:class="center-block"}