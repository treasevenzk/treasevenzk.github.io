---
layout:     post
title:      FamilySeer 2023
subtitle:   Exploiting Subgraph Similarities for Efficient Auto-tuning of Tensor Programs
date:       2025-02-21
author:     Treaseven
header-img: img/bg18.jpg
catalog: true
tags:
    - Performance Optimization
    - Auto-tuning
    - Subgraph Similarity
---


### Motivation
- 忽略子图集群的相似性
- 浪费时间在没有意义的子图上

### Design Overview

<img width="1000" height="400" src="../img/post-familyseer-design.png"/>

***identifying similar subgraphs***

子图的静态分析方法：根据核心算子
- 核心算子能融合其他算子并形成子图
- 核心算子占据融合子图的主要时间

***foresee tuning***

<img width="500" height="600" src="../img/post-familyseer-algorithm.png"/>


***multi-GPU acceleration***


### Evaluation


<img width="1000" height="500" src="../img/post-familyseer-comparison.png"/>


<img width="1000" height="300" src="../img/post-familyseer-inference-comparison.png"/>


### Reference
[Exploiting Subgraph Similarities for Efficient Auto-tuning of Tensor Programs](https://dl.acm.org/doi/pdf/10.1145/3605573.3605596)