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
- models with more layers usually generate more intermediate results, thus increasing the memory/cache pressure
- deep models usually have an insufficient amount of computations in each layer, thus degrading the processor's utilization, particularly for GPUs

### Classification of DNN Operators and Fusion Opportunity Analysis

***DNN Operators Classification***

<img width="1000" height="200" src="../img/post-dnnfusion-classification.png"/>

***Fusion Opportunity Analysis***

<img width="500" height="400" src="../img/post-dnnfusion-mapping-type.png"/>


### DNNFusion's Design

<img width="500" height="400" src="../img/post-dnnfusion-overview.png"/>

***Mathematical-Property-Based Graph Rewriting***

<img width="1000" height="350" src="../img/post-dnnfusion-example.png"/>

<img width="1000" height="450" src="../img/post-dnnfusion-graph-rewriting.png"/>

***Light-Weight Profile-Driven Fusion Plan Exploration***<br>
- overall idea<br>
(1) DNNFusion selects the starting operators from our ECG to restrict the search space <br>
(2) starting with these seed operators, DNNFusion explores fusion opportunities along the seed opeator's successors and predecessors, respectively <br>
(3) DNNFusion creates fusion plans based on an approach that combines machine-independent mapping type analysis and a profiling result database <br>

- Fusion Plan Generation Algorithm <br>
(1) Fusion seed operator selection <br>
(2) Propagated exploration along seed's successors <br>
(3) Propagated exploration along seed's predecessors <br>

<img width="500" height="350" src="../img/post-dnnfusion-fusion-plan.png"/>

<img width="500" height="640" src="../img/post-dnnfusion-fusion-plan-generation.png"/>

***Fusion Code Generation and Optimizations***<br>

- Fusion Code Generation <br>
<img width="500" height="280" src="../img/post-dnnfusion-code-generation.png"/>