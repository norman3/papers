---
layout: page
group: "synthetic_gradients"
title: "Decoupled Neural Interfaces using Synthetic Gradients"
link_url: http://arxiv.org/abs/1608.05343
---

### Introduction

- 방향성 신경망(directed neural network)은 다음의 step을 가짐.
    - 입력 데이터를 입력받아 순 방향으로 데이터를 진행하면 계산.
    - 정의된 loss 함수로 나온 생성값을 역방향으로 전파. (backprop)
    
- 이 과정에 여러 가지 Locking이 생겨남.
    - (1) *Forward Locking* : 이전 노드에 입력 데이터가 들어오기 전까지는 작업을 시작할 수 없다.
    - (2) *Update Locking* : forward 과정이 끝나기 전까지는 작업을 시작할 수 없다. (예로 backprop)
    - (3) *Backwards Locking* : forward와 backward 과정이 끝나기 전까진 작업을 시작할 수 없다.
    
- 위와 같은 제약으로 인해 신경망은 순차적으로 동기적 방식으로 진행된다.
- 학습이 이러한 과정이 당연해 보이지만 각 레이어에 비동기 방식을 도입하고 싶거나 분산 환경등을 고려하게 되면 이러한 제약이 문제가 된다.
- 이 논문의 목표는 위에서 설명한 모든 Locking 을 해결하는 것은 아니고 backprop 과정 중 발생하는 update locking을 제거하고자 하는 것.
- 이를 위해 레이어 \\(i\\) 의 weight \\(\theta_i\\) 를 backprop을 통해 업데이트 할 때 근사값을 사용하게 된다.
    - backprop 을 제거하는 효과를 가진다.

$$\frac{\partial E}{\partial w_i} = \frac{\partial E}{\partial a_i} \frac{\partial a_i}{\partial z_i} \frac{\partial z_i}{\partial w_i}$$

$$a_i = \sum{w_i x_i}$$

$$z_i = h(a_i)$$

$$\frac{\partial E}{\partial w_i} = h'(a_i)$$

$$\frac{\partial E}{\partial w_i} = f_{Bprop}\left( (h_i, x_i, y_i, w_i),(h_{i+1}, x_{i+1}, y_{i+1}, w_{i+1}),...\right) \frac{\partial h_i}{\partial w_i} \simeq \hat{f}_{Bprop}(h_i)\frac{\partial h_i}{\partial w_i}$$
