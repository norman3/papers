---
layout: page
group: "poor_convolution_network"
title: "Why do deep convolutional networks generalize so poorly to small image transformations?"
link_url: https://arxiv.org/pdf/1805.12177.pdf
---

### 문제 발견.

- 서비스를 위한 분류 모델을 작성 중 다음과 같은 이상 현상을 확인함.
    - 사용된 backbone 은 resnet-50 모델
    - 모든 과정은 동일하고 이미지 전처리 방식만 다름.
    - 왼쪽은 `tf.image.resize_bilinear` (1.4 version), 오른쪽은 다른 이미지 Tool 의 resize 함수.
        - 왼쪽의 분류 결과는 높은 확률 값으로 '가방'이 나옴.
        - 오른쪽의 분류 결과는 높은 확률 값으로 '옷'이 나옴.
    - 그런데 `resize` 결과만 봤을 때에는 tf resize 함수의 품질은 별로다. (오히려 3rd party resize 함수가 더 좋다.)
    - Overfitting 이라고 생각하기에는 너무 석연치 않은 결과.
    
![figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" }


### Why do deep convolutional networks generalize so poorly to small image transformations?

- CNN은 일반적으로 이미지 변환 및 변형에 강한 모델로 알려져있음. (즉, 일반화에 강함)
- 가장 흔히 사용되고 있는 CNN 모델을 가지고 실험을 해봤더니 실제로는 그렇지 않더라.
    - `VGG16`, `ResNet50`, `InceptionResNetV2`
    - 오히려 분류에 좋은 성능을 내는 깊은 망을 가진 네트워크들에서 이 현상이 심하다.
- 즉, CNN 모델은 통계적 편향(bias)가 존재하며 인간(human)에 비해 형편없는 일반화 능력을 가지고 있다고 할 수 있다.

### Introduction

- 우리가 CNN을 대하는 태도는 이미 superhuman 급. (객체 인식 영역)
 convolution 과 pool 연산을 통해 image translations, scalings, small deformations 에 강건한 모델이라 여기고 있음.
 
