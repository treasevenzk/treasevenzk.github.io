---
layout:     post
title:      Orojenesis ISCA 2024
subtitle:   Mind the Gap Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms
date:       2025-01-08
author:     Treaseven
header-img: img/bg20.jpg
catalog: true
tags:
    - Deep Learning
    - Adaptive Systems
    - Program Auto-tuning
---


### Motivation
- data movement is sensitive to the reuse that can be exploited by an architecture's memory hierarchy
- data movement is sensitive to the specific implementation of an algorithm

### Orojenesis
***Orojenesis Methodology***
- Snowcat architecture

<img width="500" height="300" src="../img/post-orojenesis-snowcat.png"/>

- Tool Flow

<img width="500" height="300" src="../img/post-orojenesis-flow.png"/>


<img width="500" height="300" src="../img/post-orojenesis-buffer-size.png"/>

- Extrapolating Orojenesis bounds
(1) Multi-level Memory Hierarchy
(2) Parallel Architecture
(3) Constrained Mapspaces

***Derivation Models***
- Attainable Operational Intensity Model
- Attainable Performance Model

### Single-einsum Bounds Analysis
- Matrix Multiplication
- Convolution
- Batched Matrix Multiplication

### Orojenesis Fusion
