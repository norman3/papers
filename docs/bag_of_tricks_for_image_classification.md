---
layout: page
group: "bag_of_trick_for_image"
title: "Bag of Tricks for Image Classification w/ CNN."
link_url: http://arxiv.org/abs/1812.01187
---

## Introduction

- 이미지 분류 영역에서는 2012년 Alexnet을 시작으로 CNN 방식이 표준 방식으로 자리잡음
- 이후로 새로운 아키텍쳐들이 쏟아져 나왔다.
    - VGG, NiN, Inception, ResNet, DenseNet, NASNet
    - 모두 ImageNet의 정확도를 계속 올려놓았다.
- 그런데 이러한 품질 성능 개선이 단지 모델만 바꾸는 과정으로 이루어진 것인가? => "No"
- 학습 방법을 변화시켜 얻어진 결과도 많다.
    - Loss 함수를 변경해서.
    - 데이터 입력 전처리를 변경해서.
    - 최적화(optimazation) 방법을 변경해서
- 지난 십여년간 이러한 방향으로 엄청나게 많은 기법들이 제안되었지만... 사실 그렇게 많은 주목을 받지는 못했다.
    - 게다가 정리도 잘 안되어 있어서 그냥 소스 코드 상에서만 존재하는 trick 도 많았다.
- 여기서는 이런걸 모아본다. (trick 모으기 정도로 해두자)
- 이후에 살펴보겠지만 이런 꼼수(trick)를 적용하면 성능이 올라간다.
- 하지만 여러 꼼수를 다양한 모델에서 비교하기 어려우니 ResNet-50 모델을 기준으로 이를 적용해보고 성능을 비교해본다.
    - 물론 이런 방식이 Inception.V3 와 MobileNet 에서도 동일하게 적용됨을 확인하였다.

![Table.1]({{ site.baseurl }}/images/{{ page.group }}/t01.png){:class="center-block" height="300" }

### Outline

- 평가를 시작하기 앞서 가장 먼저 baseline 을 구성한다. (Section.2)
- 다음으로 논의된 trick 들을 이 모델에 적용해본다. (Section.3)
- 3개의 minor한 모델 변화를 시도하고 품질을 확인해본다. (Section.4)
- 4개의 추가적인 정제 작업을 시도해보고 이를 논의한다. (Section.5)
- 더 효율적인 방법은 없을지 고민해본다. (Section.6)

## 학습 방법 (Training Procedures)

- 기본적인 학습 방법은 Alogrithm.1 에 정리해 두었다.

![Algorithm.1]({{ site.baseurl }}/images/{{ page.group }}/a01.png){:class="center-block" height="300" }

- 뭐, 특별한 것은 없다. (전형적인 CNN 학습 구조)
- 사실 이러한 알고리즘을 구현하는데 있서 다양한 hyper-parameter가 활용될 수 있다.
    - 즉, 서로 다른 방식으로 구현될 수 있다는 이야기이다.
    - Section.1 에서는 Algorithm.1 에 대한 실제 구현 방식을 서술한다.


### 1. Baseline Training Procedure

- 여기서는 다음의 방식으로 ResNet을 구현하고 이를 baseline으로 삼는다.
- 참고로 학습(training)과 평가(validation)를 위한 전처리(preprocessing)는 서로 다르게 처리된다.
- 학습(tranining) 과정은 매 스탭대로 진행된다.
    - 1. 랜덤 샘플링을 통해 이미지를 추출하고 이를 32bit float 타입 객체로 변환한다. (\\([0, 255]\\)
    - 2. Aspect-ratio 를 \\([3/4, 4/3]\\) 로 유지 랜덤 크롭(crop)

(작성중)