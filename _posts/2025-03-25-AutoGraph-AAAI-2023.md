---
layout:     post
title:      MetaFlow MLSys 2019
subtitle:   Optimizing DNN Computation with Relaxed Graph Substitutions
date:       2025-03-25
author:     Treaseven
header-img: img/bg2.jpg
catalog: true
tags:
    - graph substitution
---

### Motivation
图中节点的代价就是相应算子在GPU上的运行时间，整个图的代价就是所有节点之和，这种评价尺度忽略了内核并行执行的场景，会引导优化进入错误的方向


### AutoGraph

<img width="500" height="400" src="../img/post-autograph-flow.png"/>

***flow-based graph partition***

<img width="500" height="350" src="../img/post-autograph-partition.png"/>


***cost-based graph optimization***
- backtracking search via mixed critical path cost
- dp-based optimized solution search

$$
\begin{align}
C_E &= \alpha \sum_{v \in V_C} cost(v) + \sum_{v \in V} cost(v) \\
&= (1 + \alpha) \sum_{v \in V_C} cost(v) + \sum_{v \in V-V_C} cost(v).
\end{align}
$$

***on-board verification***


***overall optimization flow***


<img width="500" height="400" src="../img/post-autograph-optimization-flow.png"/>


### Evaluation

<img width="800" height="300" src="../img/post-autograph-inference-results.png"/>


<img width="500" height="300" src="../img/post-autograph-performance.png"/>



### Reference
[AutoGraph: Optimizing DNN Computation Graph for Parallel GPU Kernel Execution](https://ojs.aaai.org/index.php/AAAI/article/view/26343)