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
- residual 을 넣지 않고 일반적인 *stacked LSTM* 을 쌓아 사용하면 학습 속도가 너무 느리게 된다.
    - 게다가 잘 알고 있는대로 *exploding* 과 *vanishing gradient* 문제가 발생한다.
- 경험상 번역 모델에 기본적인 *LSTM* 모델을 사용하면 4개 층까지는 잘 동작하지만 6개부터는 문제가 발생하고 8개는 제대로 동작하지 못한다.

![figure.3]({{ site.baseurl }}/images/{{ page.group }}/f03.png){:class="center-block" height="260px"}


- 일반적인 *LSTM* 모델의 수식은 다음과 같다.

$${\bf c}_{t}^{i}, {\bf m}_{t}^{i} = LSTM_i({\bf c}_{t-1}^{i}, {\bf m}_{t-1}^{i}, {\bf x}_{t}^{i-1};{\bf W}^{i}) \qquad{(5)}$$

$${\bf x}_{t}^{i} = {\bf m}_{t}^{i} \qquad{(5)}$$

$${\bf c}_{t}^{i+1}, {\bf m}_{t}^{i+1} = LSTM_{i+1}({\bf c}_{t-1}^{i+1}, {\bf m}_{t-1}^{i+1}, {\bf x}_{t}^{i};{\bf W}^{i+1}) \quad{(5)}$$

- 이 때 \\({\bf x}\_{t}^{i}\\) 는 \\(LSTM\\) 의 스텝 \\(t\\) 일 때의 입력 값으로 사용된다.
- \\({\bf m}\_{t}^{i}\\) 와 \\({\bf c}\_{t}^{i}\\) 는 각각 \\(LSTM\\) 의 Hidden 상태와 메모리 값을 의미한다.
- 여기에 redidual 을 추가한 모델은 다음과 같다.

$${\bf c}_{t}^{i}, {\bf m}_{t}^{i} = LSTM_i({\bf c}_{t-1}^{i}, {\bf m}_{t-1}^{i}, {\bf x}_{t}^{i-1};{\bf W}^{i}) \qquad{(5)}$$

$${\bf x}_{t}^{i} = {\bf m}_{t}^{i} + {\bf x}_{t}^{i-1} \qquad{(5)}$$

$${\bf c}_{t}^{i+1}, {\bf m}_{t}^{i+1} = LSTM_{i+1}({\bf c}_{t-1}^{i+1}, {\bf m}_{t-1}^{i+1}, {\bf x}_{t}^{i};{\bf W}^{i+1}) \quad{(5)}$$

- 아주 간단한 변경만으로도 엄청난 효과를 낸다. 
- 실험 결과 8 레이어의 \\(LSTM\\) 이 가장 효과가 좋았다. (*encoder*, *decoder* 각각)

### Bi-directional Encoder for First Layer

- 번역 시스템에서는 번역을 제대로 하기 위해 입력 쪽 데이터를 양 방향에서 살펴보아야 할 필요가 있다.
    - 보통 입력 쪽 언어의 정보는 왼쪽에서 오른쪽으로 이동되는 것이 보통인데 이게 출력 쪽 언어에서는 서로 분할되어 다른 위치에 등장해야 하는 경우도 많다.
    - 컨텍스트를 충분히 살펴보기 위해 *encoder* 단계에서는 *bi-directional* *RNN* 을 사용하게 된다.
    - 아래 그림을 참고하도록 하자.
    
![figure.4]({{ site.baseurl }}/images/{{ page.group }}/f04.png){:class="center-block" height="350px"} 

### Model Parallelism

- 모델이 복잡하기 때문에 병렬 모델을 사용할 수 밖에 없다.
    - 모델 병렬화(model parallelism)와 데이터 병렬화(data parallelism)를 모두 사용한다.
- 데이터 병렬화를 위해 사용한 방식은 Downpour SGD 이다. 
    - \\(n\\) 개로 복사된 장비가 동일한 모델 파라미터를 공유한다.
    - 그리고 각각 독립적으로 파라미터를 업데이트한다. (asynchronously update)
- 실험에서는 약 10개의 장비( \\(n=10\\) )로 실험하였다.
    - 모든 장비는 \\(m\\) 개의 문장을 *mini-batch* 로 사용한다.
    - 실험에서는 주로 \\(m=128\\) 을 사용하였다.
- 추가로 모델 병렬화 (model parallelism)을 사용해서 성능을 더 올린다.
    - 다중 GPU 환경으로 모델을 구성한다. (장비당 8개의 GPU를 사용한다.)
    - 각 레이어 층마다 서로 다른 GPU를 할당하여 처리하게 된다.
    - 이렇게 하면 \\(i\\) 번째 레이어의 작업이 모두 종료되기 전에 \\(i+1\\) 번째 작업을 진행할 수 있다.
- softmax 레이어도 분할하여 사용한다.

- 현재 bi-directional LSTM 은 첫번재 레이어에만 적용한다.
    - 병렬화를 위해 최대한 효율적으로 구성하기 위해서이다.

## Segmentation Approaches

- NMT 방식은 고정된 단어 사전 집합을 사용하게 된다.
- 하지만 이런 방식은 단어 사전에 포함되지 않은 단어가 등장했을 때 처리가 애매하다는 문제가 있다. (OOV : Ouf of Vocabulary)
- 이를 해결하기 위한 가장 쉬운 방식은 *copy* 모델이다.
    - 해석이 되지 않는 단어는 그냥 입력으로부터 출력으로 전달해버리는 방식.
- 우리는 이를 좀 더 개선하기 위해 *sub-word unit* 을 만드는 방식을 추가한다.
- 이는 *work/character* 방식을 혼합한 방식이라 생각하면 된다.

### Wordpiece Model

- OOV 문제를 해결하기 위해 *sub-word unit* 방식의 WPM (wordpiece model)을 개발함.
- 이는 일본어, 한국어 (응?) 세그멘테이션(segmentation) 문제를 해결하기 위해 개발한 방법을 응용한 것이다.
- 문자 단위의 시퀀스에서 분할 가능한 단어 별로 나누어내는 과정이 들어간다. (우리는 한국 사람이니까 형태소 분석을 생각하자.)
- 아래 예제를 보면 대충 어떤 것인지 짐작이 갈 것이다.
    - **Word:** Jet makers feud over seat width with big orders at stake
    - **wordpieces:** _J et _makers _fe ud _over _seat _width _with _big _orders _at _stake
    
- 보면 알겠지만 "Jet" 이라는 단어는 "_J" 과 "et" 로 나누어지게 된다.
- 마찬가지로 "feud" 는 "_fe" 와 "ud" 로 나누어진다.
- "_" 는 시작 문자라는 의미로 붙인다.

