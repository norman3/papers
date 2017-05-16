---
layout: page
group: "fairseq"
title: "Convolutional Sequence to Sequence Learning"
link_url: "https://arxiv.org/abs/1705.03122"    
---

페이스북(facebook)이 공개한 Convoutional Seq2Seq 모델.

## Source Code

- 코드는 [여기](https://github.com/facebookresearch/fairseq){:target="blank"}에서 구할 수 있다. (루아코드임)

## Introduction

- Seq2Seq 모델은 기계번역 영역에서 매우 성공적인 모델이었음.
- 2014년 NIPS에서 Sutskever가 이를 발표. 엄청난 인기를 모음.
    - Sutskever가 최초의 Seq2Seq 발명자로 알고 있는 사람이 많은데 실은 조경현 교수가 먼저 발표한 모델. ([링크](https://arxiv.org/abs/1406.1078){:target="blank"})
- 현재 Seq2Seq에서 일반적으로 사용하는 모델은 Bi-directional RNN 모델.
    - 여기에 soft-attention 을 추가하면 금상첨화.
- 반면 CNN 모델을 이용하여 이런 시도를 하던 사람도 있긴 있었음.
    - 고정된(fixed) 입력 크기를 가져야 한다는 단점이 있진 하지만 (RNN에 비해) 생각보다 CNN의 장점도 많다.
        - 더 많은 stack을 쌓을 수 있다.
        - 이전 step에 영향을 받는 구조가 아니므로 병렬화에 유리하다.
- Multi-layer CNN은 hierarchical representation 을 갖는다.
    - 이런 구조는 RNN 방식보다 정보를 더 오랫 시간동안 유지할 수 있는 힘의 원천.
- 이 논문에서는 CNN 만을 활용하여 Seq2Seq 연산을 어떻게 제공하는지를 다룸.

## Seq2Seq with RNN

- 논문에서는 아주 간단하게 RNN을 활용한 Seq2Seq를 소개함. 하지만 여기서는 아주 간단하게만 보자.
- 어쨌거나 최근 Sequence Task의 기본 base라 할 수 있을만큼 대세인 방식.
- 관련 이론으로 RNN계열로 GRU나 LSTM 정도만 알고 있으면 될 듯 하다.
- Attention을 추가로 알고 있으면 좋다.
- 보통 최종 출력되는 output 에 대한 식은 다음과 같이 정의된다.

$$p(y_{i+1} | y_1, ...,y_i, {\bf x}) = softmax(W_o h_{i+1} + b_o) \in {\mathbf R}^V$$

- 여기에 attention 모델을 추가하면 다음과 같은 식을 사용하게 된다.

$$d_i = W_d h_i + b_d + g_i \qquad\qquad a_{ij} = \frac{\exp(d_i^Tz_j)}{\sum_{t=1}^m{\exp(d_i^Tz_t)}} \qquad\qquad c_i = \sum_{j=1}^m a_{ij}z_j$$

- 논문에서 생략한게 너무 많은 거 같다. 간단하게 적어보자.
    - \\(c\_i\\) 는 decoder 에 들어가는 입력 값. 결국 \\(a\_{i:}\\) 에 많은 영향을 받게 될 것이다.
    - \\(z\_j\\) 는 encoder 의 출력값. encoder 의 max 크기에 영향을 받게 된다. (\\(m\\))
    - \\(d\_i\\) 는 Seq 연산 중에 다음 attention 영역 계산시 사용되는 요소이다.
        - 식을 잘보자. 모든 \\(i\\) 와 \\(j\\) 에 대해 연산을 해야 한다. (\\(j\\) 는 encoder 길이. \\(i\\)는 decoder 길이가 된다.)
        - 출력에 대한 seq 정보를 얻어 다음에 바라봐야 할 attention 영역을 만들어낼 때 사용되는 값으로,
        - 일단 \\(W\_d h\_i + b\_d\\) 는 \\(g\_i\\) 와의 크기를 맞추기 위한 선형 변환이고,
        - \\(g\_i\\) 는 이전 decoder 의 출력값이다.
        - 결국 attention \\(a\_{ij}\\) 는 encoder 의 출력값에 대해 실제 decoder 의 이전 출력 seq 를 따라가면서 집중해야 할 영역을 찾아내는 과정이라는 것.
        - 설명이 어렵지만 attention 을 한번이라도 살펴봤던 사람이라면 대충 느낌적 느낌은 떠오를 것이다.
        

## Convolutional Architecture

- 표기
    - 입력 문자열 : \\({\bf x} = (x\_1, ...,x\_m)\\)
    - 입력에 대한 Embedding 벡터 : \\({\bf w} = (w\_1, ...,w\_m)\\)
    - 입력에 대한 Position 벡터 : \\({\bf p} = (p\_1, ...,p\_m)\\)
    - 실제 입력 벡터 : \\({\bf e} = (w\_1, p\_1, ...,w\_m, p\_m)\\)
    - Decoder로부터 얻어지는 Feed-back 벡터 : \\({\bf g} = (g\_1, ...,g\_m)\\)
    - Encoder 출력 벡터 : \\({\bf z^l} = (z\_1^l, ...,z\_m^l)\\)
    - Decoder 출력 벡터 : \\({\bf h^l} = (h\_1^l, ...,h\_m^l)\\)
    - Convolution 커널의 크기 : \\(k\\) 


![figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" height="550px"}

- Convolution
    - 내맘대로 표기법. (논문의 표기가 좀 이상하다.)
    - word embedding \\(e\_i^{(d)} = w\_i^{(d1)} + p\_i^{(d2)}\\). (단, \\(d=d1+d2\\))
    - input block \\(b\_k^{(kd)} = \\{e\_1^{(d)},...,e\_k^{(d)}) \\}\\) 
    - convolution 출력 \\(d\_k^1 = W^{(2d \times kd)} b\_k^{(kd)}\\)

