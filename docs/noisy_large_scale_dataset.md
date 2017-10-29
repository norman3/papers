---
layout: page
group: "noisy_large_scale_dataset"
title: "Learning From Noisy Large-Scale Datasets With Minimal Supervision"
link_url: https://arxiv.org/abs/1701.01619
---

## OpenImage

- 올해 구글이 공개한 약 900만개의 이미지 데이터.
    - 현재 약 6000개의 클래스를 제공한다.
- 경로 : https://github.com/openimages/dataset
- 실제 이미지를 제공하는 것은 아니고 사용 가능한 이미지 URL이 기술된 Text 파일을 제공한다.
    - 따라서 이미지를 사용하고 싶으면 직접 다운 받아야 한다.
    - 시간이 없다. URL Missing 비율이 계속 올라간다.
    - 토렌트를 활용하기도 하는데 전체 이미지 집합은 아니다.
- 이미지는 Multi-Label 로 기술되어 있으며 2가지 타입으로 나뉜다.
    - Machine Generated Labels : 기계가 부여한 레이블
    - Human Verified Labels : 사용자가 검정한 레이블.
- 추가로 약 600 클래스의 Bounding-Box도 함께 제공하므로 Object Detection 연구자도 살펴볼만 하다.

## Introduction

- 대규모 데이터를 학습하면 CNN 성능이 더 좋아진다는 것은 이미 알려져 있음.
- 그런데 왜 여전히 5년전에 나온 ImageNet 데이터를 여전히 활용하고 있을까?
    - 이보다 더 클린한 이미지 데이터 집합을 얻을 수 없기 때문.
    - 학습 데이터 확보가 병목이 되는 상황. (대규모 클린(clean) 데이터의 필요성)
- 어차피 힘들다는 것을 알고 있으니 다른 방안들을 고민한다.
    - unsupervised learning
    - self-supervised learning
    - learning noisy annotations
- 그리고 이러한 방식들은 모두 주어진 데이터가 에러를 많이 포함(noisy)하고 있다는 것을 전제로 한다.
- 따라서 말들은 이렇게 하지만 결국 semi-supervied 형태의 학습을 진행하게 된다.
    - Noisy를 잘 걸러내어 정수를 뽑아내자.
- 이 논문도 별반 다르지 않다.
    - 소량의 정제 데이터 (small amount of clean annotaions)
    - 대량의 Noisy 데이터
    - "이 둘을 잘 활용해서 더 좋은 모델을 만들자."

- 일반적인 접근 방법.
    - 먼저 Noisy한 데이터 집합으로 학습 후, Clean 데이터를 이용해서 Fine-Tunning
    - 근데 생각보다 잘 안된다. (Clean Data의 정보 활용 비율이 매우 낮다.-고 주장)
    
- 본 논문의 접근 방식
    - 이미지를 표현하는 모든 컨셉(concept)을 표기하는 방식으로 이미지 분류를 다룸.


### 레이블(label) 분류 접근 방식

- 많은 분류 문제에서 타겟 레이블 (Label)은 서로 독립적이라고 가정함.
    - 학습 데이터를 이에 맞춘다면 타당한 가정임.
    - 하지만 Noisy 데이터에서는 이렇지 않다.
        - 많은 class가 서로 연관 관계를 가지게 됨. (상하/포함 관계 등)

![figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" height="400px"}

- (그림1)
    - 실제 내부적으로는 annotation 사이의 relation을 학습하게 된다.
    - 녹색은 강한 양의 상관 관계. 적색은 강한 음의 상관관계.

- 목표
    - Noisy 데이터로부터 다중 레이블 분류기(multi-label image classifier)를 만들고 싶다.
    - 부가적으로 Noisy 데이터의 정제 버전을 생성할 수 있다.

- 이미지와 레이블의 상관 관계.
    - 결국 우리가 필요로 하는 모듈은 Noisy Label 사이의 상관관계를 파악해서 좋은 Label을 추출하는 것.
    - 하지만 여기에는 이미지가 반드시 필요하다.
        - 예를 들어 '코코넛' 과 같은 annotation은 중의적 의미를 포함할 수 있다. (음료수 or 실제 과일)

- 평가 방법
    - Open Image 평가 방법을 따름.
    - 그리고 실제 해보니 우리 방식이 짱 좋다.
    - 게다가 Clean Dataset 크기가 작은 경우 오히려 Noisy 만을 가지고 학습한 모델보다 추가로 Fine-Tunning 한 것이 성능이 더 나쁘다.

### 논문의 공헌 (Contribution)

1. Multi-Label Image Classifier 를 위한 Semi-supervied 학습 프레임 워크 소개
2. 최근 출시된 Open-Image 데이터의 벤치 마크 제공
3. 일반적으로 알려져있는 Fine-Tunning 기법보다 더 효과적인 Tunning 방법 제안.

## Related Work

- 기존의 방식은 대부분 2가지 방법을 사용함.
    - Noisy 데이터로 부터 Noisy 에 강한 알고리즘을 개발하는 방식
        - Label 을 적절히 전처리하거나 제거 등으로 강건한 모델을 생성
    - Noisy Dataset 과 Clean Dataset 을 적절히 결합하여 모델 성능을 올림.
        - 우리가 하는 일과 유사.

## Approach

- 일단 두 개의 집합으로 데이터를 구분함.

$$T = \left\{(y_i, I_i),...\right\}$$

$$V = \left\{(v_i, y_i, I_i),...\right\}$$

- 여기서 \\(T\\) 는 Noisy 데이터이고 \\(V\\) 는 Clean 데이터를 의미한다.
- \\(y\_i\\) 는 Noisy한 이미지의 레이블(label)이고 \\(v\_i\\) 는 Clean 이미지 (즉, 사람이 검수한) 레이블이다.
- 보통 \\(T\\) 의 크기는 \\(V\\) 보다 훨씬 더 크다. (more three orders)





    
