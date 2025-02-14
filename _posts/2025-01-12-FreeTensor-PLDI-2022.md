---
layout:     post
title:      FreeTensor PLDI 2022
subtitle:   FreeTensor A Free-Form DSL with Holistic Optimizations for Irregular Tensor Programs
date:       2025-01-12
author:     Treaseven
header-img: img/bg20.jpg
catalog: true
tags:
    - Tensor Computing
    - Optimizing Compilers
    - DSL
---

### Motivation
SubdivNet实现遇到的问题
- 需要将数据来回转换和复制
- 引入大量冗余计算和内存拷贝
- 大量操作仅用于重排数据，没有实际计算

FreeTensor遇到的挑战
- Optimization with the presence of dependence: 细粒度控制流使得代码生成更加困难，复杂的控制流和数据依赖关系限制潜在的代码转换优化
- Efficient automatic differentiation on complex control flows: 自动微分在复杂控制流程序上可能引入大量冗余，需设计高性能的自动微分机制


### Free-Form DSL


### Generating High Performance Code
- transform AST without breaking an allocation-freeing pair
- by limiting the life scope of a tensor to a sub-tree, most of the false dependence in dependence analysis can be eliminated


***Loop transformations***

<img width="500" height="400" src="../img/post-freetensor-loop-transformations.png"/>

***Parallelizing transformations***

<img width="500" height="400" src="../img/post-freetensor-parallelize.png"/>


***Memory transformations***


### Evaluation


<img width="1000" height="800" src="../img/post-freetensor-comparison.png"/>