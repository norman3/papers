---
layout: page
group: "gnmt"
title: "Google's Neural Machine Translation System."
link_url: https://arxiv.org/abs/1609.08144
---

- 구글이 만든 신경망 번역 시스템으로 GNMT (Google Nueral Machine Translation) 라고 부른다.
    - 당연히 end-to-end 구조이다.
    - 예전 방식인 phrase-based 번역 모델에 비해 짱 좋다.
    - 그리고 참고로 일반적인 신경망 번역 시스템을 NMT 라고 부른다.
- 물론 NMT 계열의 모델도 단점이 있다.
    - 연산량이 무지 높다.
    - 큰 모델을 사용하려면 많은 데이터가 필요하다.
- 실제로 서비스하려면 정확도와 속도가 생명.
- 구글이 사용한 GNMT 모델을 잠시 소개하자면,
    - 8개의 LSTM *encoder* 와 8 개의 LSTM *docoder* (게다가 **Attention** 모델)
    - 학습시 속도를 올리기 위해 low-precision 연산 처리.
    - 드문드문 발생하는 단어들도 잘 좀 처리해보자는 의미에서 *wordpiece* 를 사용.
        - 이런 방식은 *word* 단위 모델과 *character* 단위 모델의 중간 쯤으로 생각해도 된다.
    - 빠른 검색을 위한 beam-search 는 local-normalization 기법을 사용함.
    
## Introduction

- NMT 가 각광을 받는 이유는 당연하게도 기존 모델에 비해 아주 간단하면서도 좋은 성능을 내주기 때문.
- 하지만 만능은 아닌 것이 다음과 같은 제약사항이 존재한다.
    - (1) 느린 학습(training) 속도와 느린 추론(inference) 속도
        - 데이터가 크니 느린건 당연지사.
        - 추론 속도도 보통 phrase-based 방식보다 느린데 이는 모델 파라미터가 너무 많아 단위 연산 비용이 높기 때문.
    - (2) 드물게 등장하는 단어에 대한 부정확도
        - 아주 드물게 등장하는 단어는 추론(inference)시 사전에 없을 수 있다.
        - 이 경우 `<unk>` 등의 태그로 처리한다.
        - **Copy Model* 이란 방식으로 해결할 수 있긴 한데 그냥 모르는 단어는 번역 후에도 그냥 그대로 사용하는 것이다.
            - 이 경우에도 적절한 위치에 해당 단어가 놓아지게 된다.(alignment) 
            - 물론 이게 항상 좋은 결과를 주는 것은 아니다.
    - (3) 가끔씩 전체 입력 문장에 대해 다 번역을 하지 않는 경우가 생긴다.
        - 이 경우 번역 결과는 개판이 된다.

## Related Work
    
- NMT 이전에는 SMT (Statistical Machine Translation) 모델이 주류였음. ( [참고 동영상](https://youtu.be/cgpHy2D_5o4?list=PLO9y7hOkmmSH8_IPjBUf7msTyLIBsV56b) )
- 현실적은 구현 방법은 *Phrase-based Model* 였다. (단어들을 구 단위로 묶어 번역 처리함)
- 이 논문은 NMT 와 관련된 논문이니 예전 것들은 접어두고 NMT 에만 집중하자.

## Model Architecture

![figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" height="450px"}

- [그림.1] 만 봐도 *attention* 기능이 들어간 기본적인 *seq-to-seq* 모델임을 알 수 있다.
- *seq-to-seq* 를 잘 모르는 분들을 위해 2014 NIPS 에서 *Ilya Sutskever* 가   발표한 최초의 모델을 잠시 보자.

![figure.2]({{ site.baseurl }}/images/{{ page.group }}/f02.png){:class="center-block" height="150px"}

- - -

- 다시 GNMT 로 돌아와서...
- 이 모델은 크게 3가지 영역으로 나눌 수 있다.
    - 입력 문자열을 처리하는 *encoder* 네트워크.
    - 출력 문자열을 처리하는 *decoder* 네트워크.
    - 그리고 *attention* 네트워크.
    
- 표기법을 좀 살펴보도록 하자.
    - 볼드체 소문자는 벡터(vector)를 의미 (ex) \\({\bf v}, {\bf o}_i\\)
    - 볼드체 대문자는 행렬(matrix)을 의미 (ex) \\({\bf U}, {\bf V}_i\\)
    - 필기체 대문자는 집합(set)을 의미 (ex) \\(\mathcal{V}, \mathcal{T}\\)
    - 그냥 대문자는 문자열을 의미 (ex) \\(X, Y\\)
    - 그냥 소문자는 문자열 내에 속한 심볼을 의미 (ex) \\(x, y\\)

- \\(M\\) 개의 길이를 가지는 입력 문자열 : \\(X = \\{ x\_1, x\_2, ..., x\_M \\} \\) 
- \\(N\\) 개의 길이를 가지는 출력 문자열 : \\(Y = \\{ y\_1, y\_2, ..., y\_N \\} \\) 

$${\bf x}_1, {\bf x}_2, ..., {\bf x_M} = EncoderRNN(x_1, x_2, ..., x_M) \qquad{(1)}$$

- 이 때 \\({\bf x}\_1, {\bf x}\_2, ..., {\bf x\_M}\\) 는 고정된 크기를 가지는 벡터.
    - 이 벡터의 개수는 \\(M\\) 개라는 것을 유의하자.

$$P(Y|X) = P(Y|{\bf x}_1, {\bf x}_2, ..., {\bf x_M}) = \prod_{i=1}^{N} P(y_i | y_0, y_1, y_2, ..., y_{i-1};{\bf x}_1, {\bf x}_2, ..., {\bf x_M}) \qquad{(2)}$$

- 유심히 볼 것이 하나 있는데 바로 \\(y_0\\) 이다. 특별한 심벌로 문장의 시작을 나타내는 심볼이다.
- 실제 추론 단계에서는 결국 다음 심볼의 확률값을 계산하는 문제로 생각하면 된다.

$$P(y_i | y_0, y_1, y_2, ..., y_{i-1};{\bf x}_1, {\bf x}_2, ..., {\bf x_M}) \qquad{(3)}$$

- *decoder* 단계에서는 *RNN* (*LSTM*) 과 소프트맥스(*softmax*) 레이어를 사용한다.
    - *softmax* 를 이용하여 출력할 단어를 결정하게 된다.
- 보통 *LSTM* 이 깊어질수록 (레이어를 올릴수록) 성능이 좋다는 보고가 있는데 여기서도 마찬가지였다.

- *attention* 모델은 사실 [이 논문](https://arxiv.org/pdf/1409.0473v7.pdf) 을 따라한 것이다.

$$s_t = AttentionFunction({\bf y}_{i-1}, {\bf x}_t)\;\;\forall{t},\;\;1 \le t \le M  \qquad{(4)}$$

$$p_t =\frac{\exp{(s_t)}}{\sum_{t=1}^{M} \exp{(s_t)}}\;\;\forall{t},\;\;1 \le t \le M  \qquad{(4)}$$

$${\bf a}_t = \sum_{t=1}^{M} p_t \cdot {\bf x}_t \qquad{(4)}$$


### Residual Connections

- 정확도를 올리고자 residual 노드를 넣는다. (*LSTM* stack에다가)


