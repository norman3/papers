---
layout: page
group: "deepir"
title: "Deep Image Retrieval"
link_url: https://arxiv.org/abs/1610.07940
---

- **목표** : 이미지를 질의로 받아 유사한 이미지를 검색하자.

- 논문 : [Deep Image Retrieval](http://www.europe.naverlabs.com/Research/Computer-Vision/Learning-Visual-Representations/Deep-Image-Retrieval){:target="_blank"}
    - 'XRCE Research Europe' 에서 'Naver Labs Europe' 이 된 회사에서 나온 논문

## 소개

- Instance-level image retrieval
    - 간단하게 생각하면 이미지를 넣어 유사한 이미지를 반환해주는 검색
    - 보통 매우 큰 이미지 집합 내에서 검색을 해준다.
    - 웹 환경에서 사용되거나 사용자 앨범 등의 이미지에서 검색을 제공하는 등 여러 응용 예제가 있다.
- CNN 응용 예제들.
    - CNN을 이용하여 유사 이미지를 찾고자 하는 시도를 많이 했음.
        - ImageNet 으로 학습된 분류용 pretrained Network 을 얻어다가,
        - 분류 단계 이전 CNN Feature 정보로부터 유사한 이미지를 찾는 시도가 많았음.
        - 성능은 적당.
- 논문에서는 검색 목적에 부합하는 Task로 정의하여 이 문제를 해결함.
    - 검색에 유용한 정보를 추출하기 위해 R-MAC 과 같은 discriptor 를 사용
        - 이는 CNN 계열의 discriptor로 서로 다른 스케일의 이미지 Region을 추출하여 sum-aggregate 하는 방식
        - 물론 기존의 R-MAC 이론과는 조금 다르게 사용함.
        

        

