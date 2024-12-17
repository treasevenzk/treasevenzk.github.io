---
layout:     post
title:      HUMMINGBIRD OSDI 2020
subtitle:   A Tensor Compiler for Unified Machine Learning Prediction Serving
date:       2024-12-14
author:     Treaseven
header-img: img/bg20.jpg
catalog: true
tags:
    - Tensor Operations
    - Machine Learning
    - Model Scoring
---

### Contributions
(1) Can traditional ML operators be translated to tensor computations?
(2) Can the resulting computations in tensor space be competitive with the imperative alternatives we get as input?
(3) Can HB help in reducing software complexity and improving model protablility?

### System Overview
#### System Architecture and Implementation

<img width="500" height="150" src="/img/post-hummingbird-architecture.png"/>


### Compilation
#### Compiling Tree-based Models
***Strategy 1: GEMM***

<img width="500" height="300" src="/img/post-hummingbird-gemm.png"/>

***Strategy 2: TreeTraversal (TT)***

<img width="500" height="500" src="/img/post-hummingbird-treetraversal.png"/>

***Strategy 3: PerfectTreeTraversal (PTT)***

<img width="500" height="400" src="/img/post-hummingbird-ptt.png"/>

#### Summary of Other Techniques
***Exploiting Automatic Broadcasting***

***Minimize Operator Invocations***

***Avoid Generating Large Intermediate Results***

***Fixed Length Restriction on String Features***

### Optimizations
#### Heuristics-based Strategy Selection
#### Runtime-independent Optimizations
***Feature Selection Push-Down***
***Feature Selection Injection***