---
layout: page
group: "facial_age_by_posterior"
title: "Quantifying Facial Age by Posterior of Age Comparisons"
link_url: https://arxiv.org/abs/1701.01619
---

## Concept

- 얼굴 사진만 보고 나이를 맞추는 일은 사람에게도 어려운 일이다.
- 하지만 생각외로 잘 하는 것은 두 인물 사진을 주고 누가 더 늙어(혹은 어려)보이는가를 묻는 것이다.
- 이러한 개념을 받아들여 학습이 가능한 모델을 구성해본다.
- 즉, 나이가 알려지지 않은 인물 사진이 주어졌을 때 이미 나이가 알려진 이미지를 바탕으로 나이 확률에 대한 Posterior를 구하는 문제로 전환한다.

![figure.1]({{ site.baseurl }}/images/{{ page.group }}/f01.png){:class="center-block" height="400px"}

## 나이 비교로 나이 사후분포 구하기

### 실제 사람은 어느 정도의 performance를 보이는가?

- **TEST-1** : FG-NET 데이터(이미 나이가 기술된 데이터)를 일반 사용자에게 물어보았을 경우.
    - 매우 낮은 수치로 나이를 맞춘다.
    - \\(-3 ~ +3\\) 으로 맞출 확률이  43.2% 정도
- **TEST-2** : 상대적으로 두 인물의 사진을 보고 나이 차 (A-B) 를 맞추라고 하는 경우
    - 10년 이상 차이를 에러로 보면 약 95%의 정확도
    - 5년 이상 차이를 에러로 보면 약 85%의 정확도
    
    
![figure.2]({{ site.baseurl }}/images/{{ page.group }}/f02.png){:class="center-block" height="300px"}

### 나이 사후분포

- 쿼리 이미지 : \\(I\\)
- 비교할 참조 이미지 : \\(I\_{ref}\\)
- \\(I\_{ref}\\) 의 알려진 나이 \\(k\\) 에 대해 이미지 \\(I\\) 와 비교할 랜덤 이벤트를 \\(C\_k \in \{0, 1\}\\)
    - 만약 \\(C\_k=1\\) 이면 \\(I\\) 는 \\(I\_{ref}\\) 보다 늙음.
    - 반대로 0 이면 젊음.
- 이제 likelihood를 세워보자.


$$P(C_k|a) = \qquad{(1)}$$
