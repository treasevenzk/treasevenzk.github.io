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

O[i_1, i_2, \ldots, i_M] = \mathcal{F}(I_1, I_2, \ldots, I_N)

***Schedule Space Generation***

### Reference 
[FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2022_2023/papers/ZHENG_ASPLOS_2020.pdf)