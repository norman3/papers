---
layout: page
group: "delf"
title: "Large-Scale Image Retrieval with Attentive Deep Local Features"
link_url: "https://arxiv.org/pdf/1612.06321.pdf"      
---

- 재미있는 논문을 좀 정리해야 하는데 매번 시간이 없다.

## Preface

- 논문 제목보다는 그냥 DELF 라고 부르는 CNN 모델이다.
- 포항공대 + Goolgle 콜라보의 논문이다.
- Google Vision API 에서 핵심적으로 사용되는 이미지 검색 기능이 아닐까한다. (추측이다.)
    - 논문 작성자가 구글 인턴쉽 때 만든 내용으로 보인다.
- 2017년 논문이라 요즘 기준으로는 오래된 논문이다.
    - 그런데 최근 [Google Landmark](https://www.kaggle.com/c/landmark-recognition-challenge/data) 데이터가 공개되면서 새로 갱신되었다.
    - 사실은 구글 Landmark 를 위해 만들어진 모델이었으나 최초 발표시에는 이걸 자랑할 수 없었다.
    - 구글답게 데이터를 공개해버리고 논문을 새로 갱신하였다.
- 구글 Landmark Dataset 공개는 의미가 크다.
    - 전통적으로 Image 검색 분야에서는 [Oxford](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) 와 [Paris](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) 라는 데이터가 많이 사용된다.
        - 모두 Landmark 데이터이다.
        - 하지만 학술용 데이터라 좀 많이 부족하다.
    - 딱히 평가 기준으로 내세울 데이터가 없었던지라 부족해도 그냥 쓰던 상황.
        - 이런 이유로 [Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking](http://cmp.felk.cvut.cz/revisitop/) 같은 논문도 나왔다.
    - 여기서 구글이 표준이라 할만한 새로운 Landmark 데이터를 공개해버린 것.
        - 자세한 내용은 이후에...
        
## Introduction

- 논문은 일단 광고로 시작
    - local feature 를 활용한 킹왕짱 이미지 검색 시스템
    - attention 메커니즘의 도입.
    - 교체 가능한 키 포인트 디텍터 (keypoint detector)
    - FP(false positive) 를 제거할 수 있는 능력
    - 새로운 dataset 도 함께 드려요. (google landmark dataset)
    - TensorFlow models 기본 탑재. ([링크](https://github.com/tensorflow/models/tree/master/research/delf))

- 일단 Large-scale 이미지 검색 시스템을 염두해 두고 시스템이 개발되었다.
- CNN 이 모든 이미지 연구에 한 획을 그엇지만 부족한 점이 있는 것도 사실이다.
    - 대규모 검색 시스템에 도입하기가 쉽지는 않다.
    - 대규모의 이미지는 거의 쓰레기 수준. (clutter, occlusion, variations view-point, illumination)
    - Global descriptor (즉, CNN feature) 는 이미지 부분 패치에 대한 정보를 가지지 못한다.
- 그래서 최근에는 CNN 을 사용하되 이미지 국소 영역에 대한 정보를 추출하는 local descriptor 에 대한 연구가 많아지고 있다.
    - 하지만 성능은 믿을 수 없다.
    - 대부분의 논문들이 소규모의 데이터 집합을 이용한 테스트만 수행한다.
    - 제대로 하려면 (즉 유의미한 결과를 좀 만들어내려면) 대량의 학습 데이터를 활용해야 한다.
- 이 논문의 목적은 CNN을 활용한 대규모 이미지 검색 시스템.
    - 하지만 품질을 양보하진 않을 거에요.
    - 주요한 포인트는 attention 기능이 가미된 local feature 기반의 CNN 모델.

## Related Work

- 기존의 데이터 
    - Oxford5k (5,062), Query 는 55개
    - Paris6k (6,412), Query 는 55개
    - 도대체 쓸데가 없다.
    - 그래서 종종 Flickr100k (100k) 데이터를 섞어서 학습하기도 한다.
    - Holday dataset도 조금 쓰는데 1491 개의 이미지와 500개의 Query.

- 검색 기법
    - 예전에 사용되는 근사 검색 기법은(ANN) KD-Tree 를 활용. (당연히 지금은 아무도 안쓴다.)
- local feature
    - VLAD 나 Fisher Vector 같은 전통 기법들.
- global feature
    - pretrained 된 CNN 결과 등.

## Google-Landmark Dataset

- 원래 이런 데이터를 어떻게 생성하는지에 대한 논문도 낸 적이 있다. (2009년, 논문 참고)
- 어쨌거나 Landmark 데이터를 공개한다. 앞으로 이걸로 평가를 좀 하자.
- 12,894 개의 명소가 포함. 총 1,060,709 개의 이미지. 111,036개의 Query 이미지.
- 전 세계를 대상으로 추출되었으며 GPS 정보와 연계되어 있다.

![figure.2]({{ site.baseurl }}/images/{{ page.group }}/f02.png){:class="center-block" height="150px"}
