---
layout: page
group: "fairseq"
title: "Convolutional Sequence to Sequence Learning"
link_url: "https://arxiv.org/abs/1705.03122"
---

## Source Code

- 코드는 [여기](https://github.com/facebookresearch/fairseq){:target="blank"}에서 구할 수 있다.

## Introduction

- Seq2Seq 모델은 기계번역 영역에서 매우 성공적인 모델이었음.
- 2014년 NIPS에서 Sutskever가 이를 발표. 엄청난 인기를 모음.
    - Sutskever가 최초의 Seq2Seq 발명자로 알고 있는 사람이 많은데 이거 아님.
    - 조경현 교수가 이미 발표한 내용이었음. ([링크](https://arxiv.org/abs/1406.1078){:target="blank"})
- 많이 사용하는 모델은 bi-directional RNN 모델
    - 여기에 soft-attention 을 추가하면 금상첨화
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
- RNN계열로 GRU나 LSTM 정도만 알고 있으면 될 듯 하다.

## Convolutional Architecture


