---
layout:     post
title:      FamilySeer ICPP 2023
subtitle:   Exploiting Subgraph Similarities for Efficient Auto-tuning of Tensor Programs
date:       2025-03-20
author:     Treaseven
header-img: img/bg2.jpg
catalog: true
tags:
    - Performance Optimization
    - Auto-tuning
---

### Motivation
现在的方法采用单一的代价模型忽略不同子图的相似性，错失机会来提升模型搜索质量和效率；浪费时间在没有性能提升的子图上




### FamilySeer


<img width="1000" height="400" src="../img/post-familyseer-design.png"/>


<img width="500" height="600" src="../img/post-familyseer-algorithm.png"/>


### Evaluation

<img width="1000" height="600" src="../img/post-familyseer-comparison.png"/>

<img width="1000" height="300" src="../img/post-familyseer-inference-comparison.png"/>


### Reference
[Exploiting Subgraph Similarities for Efficient Auto-tuning of Tensor Programs](https://dl.acm.org/doi/pdf/10.1145/3605573.3605596)