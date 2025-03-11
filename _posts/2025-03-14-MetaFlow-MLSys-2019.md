---
layout:     post
title:      MetaFlow MLSys 2019
subtitle:   Optimizing DNN Computation with Relaxed Graph Substitutions
date:       2025-03-14
author:     Treaseven
header-img: img/bg2.jpg
catalog: true
tags:
    - graph substitution
---

### Motivation
现有的深度学习编译器采用贪心算法来替换计算图，导致错过很多复杂优化机会


### Metaflow


<img width="500" height="260" src="../img/post-metaflow.png"/>


<img width="500" height="900" src="../img/post-metaflow-example.png"/>

***search algorithm***
- cost model
- backtracking search
- flow-based recursive graph split

<img width="500" height="600" src="../img/post-metaflow-algorithm1.png"/>


<img width="500" height="350" src="../img/post-metaflow-algorithm2.png"/>


### Evaluation


<img width="1000" height="500" src="../img/post-metaflow-performance.png"/>


<img width="1000" height="260" src="../img/post-metaflow-comparison.png"/>


<img width="500" height="400" src="../img/post-metaflow-tvm.png"/>


<img width="500" height="400" src="../img/post-metaflow-training.png"/>

### Reference
[Optimizing DNN Computation with Relaxed Graph Substitutions](https://proceedings.mlsys.org/paper_files/paper/2019/file/4dd1a7279a8cfeea2660fbc34f02a2bc-Paper.pdf)