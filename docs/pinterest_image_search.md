---
layout: page
group: "pinterest_image_search"
title: "Demystifying Core Ranking in Pinterest Image Search"
link_url: https://arxiv.org/abs/1803.09799
---

## 소개

- 검색 랭킹을 연구하는 사람들은 사용자의 관심사를 검색 결과에 반영하기 위한 노력을 해왔다.
- (여기서 잠깐) 검색 랭킹에는 Learning to rank (LTR) 방식을 많이 사용한다.
    - 검색 결과 문서를 "얼마나 잘 정렬할 것인가?" 의 문제
    - 문서(doc)가 질의(query)에 얼마나 적합한지 여부 외에도 잘 정렬이 되어있는지를 확인한다.
    - 검색 랭킹 함수를 만드는데 ML 기술들을 활용한다.
- 초기 핀터레스트에서는 단순히 텍스트에 대한 relevance만 가지고 랭킹을 매겼다. (루씬/솔라로 구현)
- 하지만 핀터레스트는 일반적인 웹 이미지 검색과는 다르다능.
    - 기본적으로 문서를 다루는 것이 아니라 pin을 다룬다.
- 도대체 사용자는 왜 핀터레스트에서 검색을 하는가?
    - 그림1.을 보면 약 60여가지의 사용자 액션이 존재한다.
        - `repin`, `click-through`, `close up`, `try it` 등등
    - 사용층에 따라 검색의 흐름이 완전히 다르다.        
    - 유연한 사용자 참여(engagement) 옵션은 사용자가 이미지 검색을 어떻게 사용하는지 이해할 수 있게 해줌.

![Figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" height="400" }

- 문제는 다양한 사용자의 행동으로 얻어진 피드백을 어떻게 통합해야 하는가이다.
    - 예를 들어 보통의 검색 서비스에서는 클릭된 결과가 클릭되지 않은 결과보다 중요하다.
    - 핀터레스트에서는 이게 꼭 정답이라고 말하기 어렵다.
    - `try in` 핀이 `close up` 핀보다 중요한 것인가?
- 또 하나의 중요한 문제는 웹 문서와 비교해볼 때 이미지에 존재하는 텍스트가 더 적고 noisy하다는 것.
    - "한장의 사진은 천 개의 단어보다 가치있다."는 말이 있긴 하지만 이미지에서 정보를 추출하는 것은 어려운 일이다.
- 마지막으로, learning to rank 와 관련된 많은 논문이 존재하기는 하지만 실무에 사용할만한 것들은 부족한 상황이다.
    - 서비스 응답 시간등을 고려해서 효율적인 랭킹 알고리즘을 사용하는 것이 중요.

- 이 논문에서는 세 가지 관점으로 문제를 다루어본다.

- **data**
    - 사용자 액션으로 부터 얻은 명시적인 피드백을 정답 레이블을 가진 학습 데이터로 통합
    - 서로 다른 랭킹 함수를 학습하기 위해 사람이 직접 판단한 데이터(relevance base)와 피드백 데이터를 병렬적으로 혼합하여 사용한다.
    - 즉, engagement-base ranking 모델과 relevance base ranking 모델을 하나의 최종 랭킹 모델로 통합한다.
- **Featurization**
    - pin에 포함된 텍스트와 비주얼 시그널을 추출하는 문제를 해결하기 위해,
    - word embedding, visual embedding, resual relevance signal, 쿼리, 사용자 의도 분석 등을 모두 사용한다.
    - 사용자가 핀터레스트에서 이미지 검색을 왜 사용하는지를 찾기 위해 다양한 feature 엔지니어링을 사용하고 사용자 분석을 수행하였다.
    - 이를 개인화 추천 기능에도 활용함.
- **Modeling**
    - 검색 품질과 응답 속도의 균형을 맞추기 위해 단계식(cascading) 랭킹 요소를 설계함
    - 단계식 코어 랭킹은간단한 랭킹 함수를 사용하여 수백만의 후보를 수천의 수준으로 낮추도록 필터링한 뒤 몇 천개 내의 pin들에 대해 강력한 랭킹 모델을 적용하여 더 좋은 품질을 만들어 낸다.
    - 단계식 코어 랭킹의 각 스테이지에서는 다양한 랭킹 모델에 대해 더 나은 모델이 무엇인지에 대한 자세한 연구를 수행.

## Engagement And Relevance Data in Pinterest Search

- 검색 결과를 평가하는 다양한 방법들이 존재함. 예를 들어
    - human relevane judgment
    - user behavioral metrics (click-through rate, repin rate, close-up rate, abandon rate etc)
- 완벽한 검색 시스템은 relevance, user-engaged 결과가 모두 좋은 시스템.
- 따라서 우리는 비교적 독립적딘 두 개의 파이프라인을 도입하여 시스템을 개발.
    - engagement data pipeline
    - human relevance judgment data pipeline

### Engagment Data

- Joachims이 click-through 데이터를 이용하여 학습하는 방법을 제시한 후 이 방식이 검색 엔진에서 LTR을 사용하는 표준이 되었음
- 핀터레스트 검색 엔진에서는 사용자 행동에 관한 데이터가 다양하기 때문에 이것을 어떻게 결합해야 하는지에 대한 고민이 많았음.
- Engagement 는 \\(<q, u, (P,T)>\\) 로 표현 가능함.
    - 사용자 쿼리는 \\(q\\), 사용자는 \\(u\\), 그리고 \\(P\\) 는 사용자가 수행한 engagement pin, \\(T\\) 는 engagement map을 의미한다.
    - map은 각 action \\(P\\) 에 대한 count 정보를 저장하고 있음.
- 앞서 그림1에서 본 것처럼 다양한 사용자 행위가 존재.
- 한가지 방법은 유저 행동 타입별로 각각의 모델을 만들고 각각의 스코어를 조절해서 합치는 방법이다. (calibration)
    - click-based, repin-based, closeup-based 랭킹 모델
    - 결론적으로 말하면 이런 방법은 실패함. (근데 이유는 설명 안함)
- 그래서 모델의 출력 결과를 합치는 방식 대신 데이터 레벨에서 합치는 방식을 사용
    - 각 유저 행동마다 가중치가 있고 유저 행동과 각 유저 행동의 가중치 합을 pin 점수 \\(l(p\|q,u)\\)를 사용. (간단하게 \\(l\_p\\)로 사용한다.)
    - \\(p \in P\\), 키워드 쿼리 \\(q\\), 사용자 \\(u\\).

$$l_p = \sum_{t\;\in\;T}{w_t c_t}\quad{(1)}$$

- \\(T\\) : engagement action
- \\(c\_t\\) : 액션 \\(t\\)에 대한 raw engagment count
- \\(w\_t\\) : 각각의 액션 유형의 양에 반비례한 값.

$$l_p = l_p\left(\frac{1}{\log{(age_p/\tau)}+1.0}+e^{\lambda pos_p}\right)\quad{(2)}$$

- \\(age\_p\\)와 \\(pos\_p\\)는 각각 핀 \\(p\\)의 age와 position.
- \\(\tau\\) 는 age 에 대한 normalized weight. \\(\lambda\\) 는 position decay를 위한 파라미터.

- Engagement 학습셋을 만드는데 어려운 점은 positive 샘플에 비해 negative 샘플이 많다.
    - Negative pruning 을 수행한다.


### Human Relevance Data

- Engagement 학습 데이터는 현재 서비스중인 랭킹 함수에 대한 Bias가 있다. (position bias 등)
- 그래서 relevance judgment 데이터도 함께 사용한다.
- 다음과 같은 형태로 평가 Tool 을 만들어 사용한다.

![Figure.2]({{ site.baseurl }}/images/{{ page.group }}/f02.png){:class="center-block" height="400" }

### Combining Engagement with Relevance

- Engagement와 Relevance의 분포가 서로 많이 다르다.

![Figure.3]({{ site.baseurl }}/images/{{ page.group }}/f03.png){:class="center-block" height="400" }

- 따라서 두 데이터 소스를 독립적으로 랭킹 함수 학습에 사용하고 이후에 모델 앙상블을 수행.

![Table.1]({{ site.baseurl }}/images/{{ page.group }}/t01.png){:class="center-block" height="400" }


    