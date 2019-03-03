---
layout: page
group: "dropblock"
title: "DropBlock: A regularization method for convolution networks"
link_url: https://arxiv.org/abs/1810.12890
---

## Introduction

- 딥러닝은 매우 많은 수의 파라미터를 가지고 있고 weight decay 와 dropout 같은 regularization 기법을 사용한다.
- 그 중 dropout 은 CNN에서 가장 먼저 성공한 기법 중 하나이다.
- 하지만 최근 CNN구조에서는 dropout 을 거의 사용하고 있지 않다.
    - 사용한다고 해도 맨 마지막 FC 레이어 정도에서만 사용한다.
- 우리는 dropout 의 약점이 랜덤하게 feature 를 drop 하는데 있는것이 아닐까 논의했다.
    - 구조상 FC에는 적당히 영향을 주게 되지만 conv 연산에는 크게 도움이 되지 않는다.
    - conv 에서는 drop을 시켜도 다음 레이어로 정보가 전달된다.
    - 그 결과 network가 overfit 된다.
- 이 논문에서는 DropBlock 기법을 제안한다.
    - dropout 보다 훨씬 성능이 좋다.

![Figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" height="400" }

### DropBlock

- DropBlock 은 dropout 과 유사한 아주 간단한 기법이다.
- dropout 과의 차이점은 독립적으로 랜덤하게 drop 하는 것이 아니라 feature 의 일정 범위를 함께 drop하는 것이다.
- DropBlock 은 주요한 2개의 파라미터로 구성된다.
    - \\(block\_size\\) : drop 할 block 의 크기
    - \\(\gamma\\) : 얼마나 많은 activation unit 을 drop할지 비율
- 실제 구현 방법은 알고리즘.1 에 서술되어 있다.

![Algorithm1]({{ site.baseurl }}/images/{{ page.group }}/a01.png){:class="center-block" height="300" }

- dropout 과 유사하게 \\(Inference\\) 시에는 DropBlock 기능을 사용하지 않는다.
    - 이는 평가시 여러 작은 net 을 앙상블하여 평균값을 취하는 것과 마찬가지의 효과를 가진다.

- \\(block_size\\) 할당하기
    - 구현에서는 \\(block_size\\) 에 고정된 상수 값을 적용한다.
        - 레이어의 feature map 크기에 상관없이 모두 동일값 사용.
    - 만약  \\(block_size=1\\) 을 사용하면 SpatialDropout 과 동일해진다.
- \\(\gamma\\) 할당하기
    - 경험에 의거하여 \\(\gamma\\)에 명시적인 값을 설정하지 않는다.
    - 초반에는 \\(\gamma\\)가 drop 하기 위한 feature 의 갯수를 조절하게 된다.

- (작업중)

