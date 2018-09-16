---
layout: page
group: "bam_and_cbam"
title: "CBAM: Convolutional Block Attention Module"
link_url: https://arxiv.org/abs/1807.06521
---

- 저자가 한국인이다. 그래서 한글 설명 자료가 있다.
    - 참고 문서 : [링크](https://blog.lunit.io/2018/08/30/bam-and-cbam-self-attention-modules-for-cnn/)
    - 사실 따로 문서를 작성할 필요도 없이 그냥 이 자료를 보면 된다.

## 이미지에서의 Attention.

- 이미지 처리(즉, CNN) 쪽에서 사용되는 Attention 는 주로 VQA 나 Captioning 분야였다.
    - [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)
- 즉, Attention은 Sequence 구조를 채용하는 모델에서 Seq. 진행 중 집중을 해야 하는 Local 영역의 Context 를 찾기(attention)위한 기법으로 사용되었다.
    - 따라서 VQA 나 Captioning 등 이미지와 문장(Sentence)을 함께 사용하는 Task 에서나 관심을 받은게 사실이다.
- 단일 이미지를 입력받는 Task 에서는 자연스럽게 Attention 을 사용할 이유가 없었음.
    - 대신 동영상은 이미지의 Seq. 로 취급될 수 있으므로 당연히 관심을 가지게 됨.
- Self-Attention 의 등장으로 Attention 의 개념이 확장됨. (?)
    - Seq. 구조를 탈피하여 무언가 집중(attent)한다는 의미로 사용된다는 느낌.
    - 전통적인 image classification / detection 에서도 적용이 되기 시작함.
    - 참고할만한 자료 : [링크](https://www.slideshare.net/WhiKwon/attention-mechanism)

## Method and Results

- 입력 : Conv feature (3-dim feature, HWC)
- 이에 대한 element 단위 attention 을 계산. (element-wise 곱)
    - sigmoid 를 사용하여 weighting 연산

$$F \otimes M(F)$$

- 이런 아이디어는 이미 RAN (Residual Attention Networks) 과 같은 논문에서 제시.
    - 단점은 연산량이 너무 많다.
    - 대신 성능 향상은 1~2 % 수준
- 비슷한 효과를 내면서 아주 가벼운 연산만을 추가 사용하는 방안을 고민.
    - 게다가 기존 모델을 그대로 사용하여 반영 가능.

## BAM (Bottlenect Attention Module)

![figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" width="600" }

- Attention 모듈은 각 네트워크의 bottlenect 영역에 위치
    - 즉, pooling 이 일어나는 위치 앞단에 BAM 모듈을 추가함.


![figure.2]({{ site.baseurl }}/images/{{ page.group }}/f02.png){:class="center-block" width="600" }

