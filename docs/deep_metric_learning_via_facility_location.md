---
layout: page
group: "deep_metric_facility_location"
title: "Deep Metric Learning via Facility Location"
link_url: http://openaccess.thecvf.com/content_cvpr_2017/papers/Song_Deep_Metric_Learning_CVPR_2017_paper.pdf
---

## 개론

- 최근 연구에 따르면 CNN 을 활용하여 이미지의 유사도를 측정하고자 하는 시도가 많았음.
- 유사도 계산을 위한 방법을 모델 안에서 직접 학습하는 것을 Metric Learning 이라고 함.
- 지금까지 다양한 방법들이 제시되어 왔음.
- 이 논문에서는 앞서 사용되었던 방법들을 고찰하고 Facility Location 이라는 방법을 제안함.

## 관련 연구

- CNN을 활용한 연구 중 semantic embedding 을 활용하는 기법들을 순서대로 확인해보자.

### Contrastive Embedding

- 샴(Siamese) 네트워크를 활용한 방법. 한 쌍(pair)으로 이루어진 데이터 집합이 필요하다. \\( \\{({\bf x}\_{i},{\bf x}\_{j}, y\_{ij})\\} \\)


$$J = \frac{1}{m} \sum_{(i,j)}^{\frac{m}{2}} y_{i,j} D_{i,j}^{2} + (1-y_{i,j})\left[\alpha - D_{i,j}\right]_{+}^{2}\qquad{(1)}$$

- 여기에서 \\(f(\cdot)\\) 는 CNN 망으로부터 얻어진 *feature* 라고 정의한다.
- 그리고 두 *feature* 에 대한 거리는 \\(D\_{i,j} = \|\|\;f({\bf x}\_{i}) - f({\bf x}\_{j})\;\|\|\\) 로 정의한다.
- \\(y\_{i,j}\\) 는 *indicator* 로 \\(y\_{i,j} \in \\{ 0, 1\\}\\) 이며, 한 쌍의 데이터가 동일한 클래스이면 1, 아니면 0 의 값을 가지게 된다.
- \\([\cdot]\_{+}\\) 는 Hinge Loss 함수를 의미한다. 즉, \\(\max(0, \cdot)\\) 과 동일한 의미가 된다.


### Triplet Embedding

 


