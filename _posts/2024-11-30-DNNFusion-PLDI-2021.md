---
layout:     post
title:      DNNFusion PLDI 2021
subtitle:   DNNFusion Accelerating Deep Neural Networks Execution with Advanced Operator Fusion
date:       2024-11-30
author:     Treaseven
header-img: img/bg26.jpg
catalog: true
tags:
    - Compiler Optimization
    - Operator Fusion
    - Deep Neural Network
---

### Motivation
- 过去的融合模式太局限没有考虑到各类算子和层连接
- 针对循环融合都是以一种低级视角看待计算
在资源受限的移动平台上高效执行更深神经网络是十分困难由于其高内存和计算要求

### DNNFusion's Design

<img width="500" height="400" src="../img/post-dnnfusion-overview.png"/>

***Mathematical-Property-Based Graph Rewriting***
优化目标：去除非必要算子、消除冗余中间数据拷贝、用高效算子来替代昂贵算子

<img width="1000" height="350" src="../img/post-dnnfusion-example.png"/>

<img width="1000" height="450" src="../img/post-dnnfusion-graph-rewriting.png"/>

***Light-Weight Profile-Driven Fusion Plan Exploration***<br>
算法核心：选定种子(这里论文提到要选有更少中间结果融合的算子作为起点，这样最终能融合更多算子，这一点有点反常规做法，常规都会偏向选择更多中间结果融合的算子)、后向传播融合、前向传播融合(该算法本质上是一个贪心算法)----这个想法能不能用到Heron里面

<img width="500" height="350" src="../img/post-dnnfusion-fusion-plan.png"/>

<img width="500" height="640" src="../img/post-dnnfusion-fusion-plan-generation.png"/>

***Fusion Code Generation and Optimizations***<br>


<img width="500" height="280" src="../img/post-dnnfusion-code-generation.png"/>


### Reference
[DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion](https://dl.acm.org/doi/pdf/10.1145/3453483.3454083)