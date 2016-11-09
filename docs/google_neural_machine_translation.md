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
    
### Introduction

- NMT 가 각광을 받는 이유는 당연하게도 기존 모델에 비해 아주 간단하면서도 좋은 성능을 내주기 때문.
- 하지만 만능은 아닌 것이 다음과 같은 제약사항이 존재한다.
    - (1) 느린 학습(training) 속도와 느린 추론(inference) 속도
        - 데이터가 크니 느린건 당연지사.
        - 추론 속도도 보통 phrase-based 방식보다 느린데 이는 모델 파라미터가 너무 많아 단위 연산 비용이 높기 때문.
    - (2) 드물게 등장하는 단어에 대한 부정확도

    - (3) 가끔씩 전체 입력 문장에 대해 모두 실패하는 번역


