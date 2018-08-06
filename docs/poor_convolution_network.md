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
    
![figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" width="400" }


### Why do deep convolutional networks generalize so poorly to small image transformations?

- CNN은 일반적으로 이미지 변환 및 변형에 강한 모델로 알려져있음. (즉, 일반화에 강함)
- 가장 흔히 사용되고 있는 CNN 모델을 가지고 실험을 해봤더니 실제로는 그렇지 않더라.
    - `VGG16`, `ResNet50`, `InceptionResNetV2`
    - 오히려 분류에 좋은 성능을 내는 깊은 망을 가진 네트워크들에서 이 현상이 심하다.
- 즉, CNN 모델은 통계적 편향(bias)가 존재하며 인간(human)에 비해 일반화 능력이 형편없다.

### Introduction

- 우리는 객체 인식 영역에 있어 CNN이 인간의 능력을 훨씬 뛰어 넘는다고 생각한다.
- 즉, convolution 과 pool 연산을 통해 image translations, scalings, small deformations 에 강건한 모델이라 여기고 있음.
- 하지만 이전의 연구들을 통해 약간의 입력 이미지 변화에도 잘못된 판단을 보이는 현상을 확인했었다.

![figure.2]({{ site.baseurl }}/images/{{ page.group }}/f02.png){:class="center-block" width="700" }

<center>Adversarial Examples that Fool both Computer Vision and Time-Limited Humans</center>
<center>https://arxiv.org/pdf/1802.08195.pdf</center>

- 물론 이러한 공격은 입력 이미지에 대한 괴랄할 변형을 가하는 작업을 진행하므로 일반적인 상황에서는 절대 이런 일이 발생하지 않을 것이라고 기대한다.
- 하지만 이 논문에서는 자연에 존재하는 실제 이미지들의 변환에 의해 CNN이 불변성을 가지는지를 확인해 볼 것이다.
    - 이중 대중적인 CNN 인 `VGG-16` , `ResNet-50` , `InceptionResNet-V2` 에 대해 확인을 해 볼 것이다.

### CNN의 실패?

![figure.3]({{ site.baseurl }}/images/{{ page.group }}/f03.png){:class="center-block" width="700" }

- 여러 입력 변화에 따른 CNN Prediction 결과의 변화 (InceptionResNet-V2) ([동영상 링크](https://www.youtube.com/watch?v=M4ys8c2NtsE&feature=youtu.be))
    - `Translation` : 이미지를 조금씩 이동
    - `Scale` : 이미지 크기를 변화
    - `Natural` : 연속된 이미지

![figure.4]({{ site.baseurl }}/images/{{ page.group }}/f04.png){:class="center-block" width="700" }

- ImageNet의 validation set에서 200개의 이미지를 랜덤하게 추출하여 translation 에 대한 실험을 수행.
- 위의 결과를 보면 약간의 translation 에도 결과가 매우 민감하다는 것을 알 수 있다.
- 그림에서 밝은 색은 올바는 레이블로 부여했을 경우를 의미하고 검은색의 경우 잘못된 결과를 부여한 경우이다.
- 놀랍게도 많은 행에서 밝은 색 -> 어두운 색으로의 전환이 급격하게 이루어진다.
- 여기서는 `jaggedness` ([dƷӕgɪdnis] '들쭉날쭉'이라는 뜻) 라는 정량적 지표를 내세워 불변성의 정도를 측정한다. 
- **jaggedness**
    - 최초 망의 결과로 top-5 내에 정답이 포함되어 있는 이미지에서 1개의 pixel을 임의로 이동
    - 이후 top-5 내에 정답 클래스가 삭제되는지를 확인
    - 실제 이 방식으로 28% 의 이미지의 클래스가 변경된다는 것을 확인.

- 한가지 특이한 사항은 망의 깊이에 따라 결과가 다르다는 것이다.
    - 망의 깊이가 깊을수록 정확도는 더 높지만 불변성에 취약하다.
    - 따라서 상대적으로 망의 길이가 짧은 `vgg` 가 `translation`에 훨씬 안정적이다.
- 참고로 각 Net 의 정확도와 크기는 다음과 같다.

![figure.5]({{ site.baseurl }}/images/{{ page.group }}/f05.png){:class="center-block" width="500" }

- 이 결과에 대한 자연스러운 고민은 혹시 `resize` 나 `inpainting` 도 연관이 있지 않을까 하는 것이다.
- 이를 알아보기 위해서 또 다른 방법으로 확인을 해본다.
    - ImageNet 에 Bounding Box 정보가 포함되어 있는데 여기에 있는 Object 를 중심으로 적당히 Cropping 을 수행한다.
    - 이렇게 모은 이미지로 다시 평가를 해본다.

![figure.6]({{ site.baseurl }}/images/{{ page.group }}/f06.png){:class="center-block" width="500" }

### Ignoring the Sampling Thereom

- CNN 에서 translation 문제가 발생한다는 것은 조금 의아하다.
- CNN 에서 convolution 연산으로 만들어지는 Representation 은 translation 이미지인 경우 마찬가지로 translation 된 Representation 이 만들어질 거라 생각되기 때문에다.
- 게다가 마지막에 Global Pooling 을 사용한다면 마지막에 pooling 작업을 수행하게 되므로 translation 에 불변이 아닐까하는 생각이 들게 된다.
    - 참고로 `ResNet50` 과 `InceptionResNetV2` 에서 사용.
- 이는 `stride` 라고 부르는 기법을 고려하지 않았기 때문.
    - sub-sampling 을 사용하는 시스템에서는 translation 불변성(invariant)을 보장할 수 없다. ([Simoncelli et al.](http://persci.mit.edu/pub_pdfs/simoncelli_shift.pdf){:target="_blank"})
    - 딥러닝의 경우 매우 많은 수의 subsampling 연산이 포함되어 있기 때문에 subsampling factor 가 매우 크다.
    - 정확히는 모르겠으나 Simoncelli 는 이러한 subsampling 의 불변성 영향도를 지표로 뽑을 수 있나보다.
        - 논문에서는 이 규칙에 따라 `InceptionResnetV2` 에 대해 45의 subsampling factor 를 가지고 있고,
        - 이해 대해 translation 불변성을 가지는 요소들은 전체 translation 가능성에 대해 \\(\frac{1}{45^2}\\) 뿐이라고 한다.
        - 정확한 계산법은 Simoncelli 논문을 봐야 할 듯.
        
