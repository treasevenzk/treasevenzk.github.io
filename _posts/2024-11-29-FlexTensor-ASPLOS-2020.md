---
layout:     post
title:      FlexTensor ASPLOS 2020
subtitle:   FlexTensor An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System
date:       2024-11-29
author:     Treaseven
header-img: img/bg27.jpg
catalog: true
tags:
    - Compiler Optimization
    - Code Generation
    - Heterogeneous Systems
---

### Motivation
with tensor-oriented data analytics is how to design a high-performance library for various tensor algorithms on heterogeneous systems
面临的挑战:
(1) 不同的调度原语组合会导致不同性能
(2) 不同的硬件也会增加复杂性

### Overview

<img width="500" height="300" src="/img/post-flextensor-overview.png"/>

### Front-end Analysis and Schedule Space
***Static Analysis***:
$O[i_1, i_2, \ldots, i_M] = \mathcal{F}(I_1, I_2, \ldots, I_N)$ <br>
spatial loops: the loops without data dependency <br>
reduce loops: the lossp have data dependency and usually run in serial <br>
statistical information $\rightarrow$ graph nodes (number of spatial loops and reduce loops、trip counts of spatial loops and reduce loops、loop orders) <br>
structural information $\rightarrow$ graph edges (number of nodes in mini-graph、number of input tensors and output tensors of each node、number of consumer nodes of each node)

***Schedule Space Generation***
(1) limit the depth of primitives combination <br>
(2) prune the parameter space <br>
(3) pre-determine certain decisions for different hardware

### Back-end Exploration and Optimization
***Exploration with Heuristics and Machine learning***:
(1) which point in H is selected as the starting point for the next step (heuristic method) <br>
(2) given the starting point p, which direction d to move along to get a new point in G (machine learning method)

***Performance Comparison***

***Optimized Schedule Implementation***:

<img width="500" height="300" src="/img/post-flextensor-algorithm.png"/>


<img width="1000" height="400" src="/img/post-flextensor-schedule-generation.png"/>


### Evaluation
#### Overall Speedups on GPUs

<img width="1000" height="400" src="/img/post-flextensor-performance.png"/>


<img width="1000" height="600" src="/img/post-flextensor-performance-2D.png"/>


<img width="1000" height="200" src="/img/post-flextensor-exploration-time.png"/>


### Thinking



### Reference 
[FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2022_2023/papers/ZHENG_ASPLOS_2020.pdf)