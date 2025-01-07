---
layout:     post
title:      SmartMem ASPLOS 2024
subtitle:   SmartMem Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile
date:       2025-01-02
author:     Treaseven
header-img: img/bg20.jpg
catalog: true
tags:
    - Mobile Devices
    - Layout Transformations
    - Transformer
---


### Design of SmartMem

#### Operator Classification and Analysis
- the performance of the computation depends upon the input layout or is independent
- the output layout is customizable

<img width="500" height="350" src="../img/post-smartmem-operator-classification.png"/>


<img width="500" height="250" src="../img/post-smartmem-operator-type-definition.png"/>


#### Layout Transformation Elimination Analysis

<img width="500" height="200" src="../img/post-smartmem-operator-combination.png"/>


<img width="500" height="300" src="../img/post-smartmem-design-decisions.png"/>

***Operator Elimination based on Index Comprehension***

<img width="500" height="250" src="../img/post-smartmem-example.png"/>

***A Reduction Dimension Based Layout Selection***
- a local layout selection for tensors associated with individual edges in the computational graphm

<img width="500" height="350" src="../img/post-smartmem-example-reduction.png"/>

#### Mapping Tensor to Texture Memory and Other Optimizations

<img width="500" height="250" src="../img/post-smartmem-sample-layouts.png"/>


<img width="500" height="300" src="../img/post-smartmem-data-access-patterns.png"/>


### Evaluation

<img width="1000" height="600" src="../img/post-smartmem-model-characterization.png"/>


<img width="1000" height="600" src="../img/post-smartmem-end-to-end-latency.png"/>


<img width="500" height="300" src="../img/post-smartmem-memory-access-count.png"/>

#### Optimization Breakdown and Analysis

<img width="500" height="250" src="../img/post-smartmem-performance-breakdown.png"/>


<img width="500" height="250" src="../img/post-smartmem-optimization-breakdown.png"/>


### Reference
[Accelerated Auto-Tuning of GPU Kernels for Tensor Computations](https://dl.acm.org/doi/pdf/10.1145/3650200.3656626)