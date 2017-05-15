---
layout: page
group: "fairseq"
title: "Convolutional Sequence to Sequence Learning"
link_url: "https://arxiv.org/abs/1705.03122"
---

## Source Code

- 코드는 [여기](https://github.com/facebookresearch/fairseq){:target="blank"}에서 구할 수 있다. (루아코드임)

## Introduction

- Seq2Seq 모델은 기계번역 영역에서 매우 성공적인 모델이었음.
- 2014년 NIPS에서 Sutskever가 이를 발표. 엄청난 인기를 모음.
    - Sutskever가 최초의 Seq2Seq 발명자로 알고 있는 사람이 많은데 이거는 거짓.
    - 조경현 교수가 먼저 발표한 내용이었다. ([링크](https://arxiv.org/abs/1406.1078){:target="blank"})
- 많이 사용하는 모델은 Bi-directional RNN 모델
    - 여기에 soft-attention 을 추가하면 금상첨화.
- 반면 CNN 모델을 이용하여 이런 시도를 하던 사람도 있었음. (흔하진 않다.)
    - 고정된(fixed) 입력 크기를 가져야 한다는 단점이 있진 하지만,
    - (RNN에 비해) 생각보다 CNN의 장점도 많다.
        - 더 많은 stack을 쌓을 수 있다.
        - 이전 step에 영향을 받는 구조가 아니므로 병렬화에 유리하다.
- Multi-layer CNN은 hierarchical representation 을 갖는다.
    - 이런 구조는 RNN 방식보다 정보를 더 오랫동안 유지할 수 있는 힘의 원천.
- 이 논문에서는 CNN만을 활용하여 Seq2Seq 연산을 어떻게 제공하는지를 다룸.

## Seq2Seq with RNN

- 따로 소개를 할 필요는 없는 듯 하여 생략한다.
- 어쨌거나 최근 Sequence Task의 기본 base라 할 수 있을만큼 대세인 방식.
- 관련 이론으로 RNN계열로 GRU나 LSTM 정도만 알고 있으면 될 듯 하다.
- Attention을 추가로 알고 있으면 좋다.

$$p(y_{i+1} | y_1, ...,y_i, {\bf x}) = softmax(W_o h_{i+1} + b_o) \in {\mathbf R}^V$$

$$d_i = W_d h_i + b_d + g_i \qquad\qquad a_{ij} = \frac{\exp(d_i^Tz_j)}{\sum_{t=1}^m{\exp(d_i^Tz_t)}} \qquad\qquad c_i = \sum_{j=1}^m a_{ij}z_j$$

## Convolutional Architecture

- CNN을 살펴보기 전에 Pooling Encoder를 먼저 살펴보자.
    - `k` 개의 관심 단어에 대해 Embedding 값을 평균내어 사용하는 방식이다.
    - 이 때 포지션 정보를 추가한다. (position embedding 을 추가한다.)

$$e_j = w_j + l_j \qquad\qquad z_j = \frac{1}{k} \sum_{t=-\lceil k/2 \rceil }^{ \lceil k/2 \rceil } e_{j+t} \qquad\qquad c_t = \sum_{j=1}^m a_{ij} e_j$$



