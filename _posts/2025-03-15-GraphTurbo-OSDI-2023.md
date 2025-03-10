---
layout:     post
title:      GraphTurbo OSDI 2023
subtitle:   Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators
date:       2025-03-15
author:     Treaseven
header-img: img/bg2.jpg
catalog: true
tags:
    - graph substitution
---

### Motivation
先前的方法没有考虑硬件架构，导致会产生更多的kernel和要求更多数据移动；生成细粒度的子图会导致错过跨层指令调度的机会；进一步导致不能充分利用更快的本地内存

### Overview of GraphTurbo

***scheduling sub-graph instances***
- collecting splitting information

<img width="500" height="300" src="../img/post-graphturbo-algorithm1.png"/>

- grouping sub-graphs


<img width="500" height="350" src="../img/post-graphturbo-algorithm2.png"/>

- ordering sub-graph instances


<img width="500" height="400" src="../img/post-graphturbo-order.png"/>


<img width="500" height="350" src="../img/post-graphturbo-algorithm3.png"/>


- inferring core binding and buffer scopes


<img width="500" height="360" src="../img/post-graphturbo-algorithm4.png"/>


- concatenating instance outputs
- generalizing the approach

***kernel generation for sub-graph instances***

<img width="500" height="400" src="../img/post-graphturbo-expand.png"/>

- loop fusion within layers
- buffer stitching across layers/blocks
- memory allocation and reuse
- across-layer instruction scheduling





### Evaluation


<img width="500" height="500" src="../img/post-graphturbo-speedup.png"/>


<img width="500" height="260" src="../img/post-graphturbo-breakdown.png"/>


<img width="500" height="230" src="../img/post-graphturbo-utilization.png"/>


### Reference
[Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators](https://www.usenix.org/system/files/osdi23-zhao.pdf)