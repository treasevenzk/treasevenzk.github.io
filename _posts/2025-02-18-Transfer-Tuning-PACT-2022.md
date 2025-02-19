---
layout:     post
title:      Transfer-Tuning 2022
subtitle:   Transfer-Tuning Reusing Auto-Schedules for Efficient Tensor Program Code Generation
date:       2025-02-18
author:     Treaseven
header-img: img/bg18.jpg
catalog: true
tags:
    - compute schedules
    - auto-tuning
    - auto-scheduling
    - tensor programs
    - tensor compilers
---

### Motivation

<img width="500" height="700" src="../img/post-transfer-tuning-example.png"/>

### Transfer-Tuning
#### Principles of Transfer-Tuning

<img width="500" height="700" src="../img/post-transfer-tuning-algorithm-1.png"/>

transfer-tuning: when we apply the schedule produced for a given kernel via auto-scheduling and apply it to a kernel other than the one the schedule was tuned for

#### Kernel Classes

<img width="500" height="200" src="../img/post-transfer-tuning-algorithm-2.png"/>

<img width="1000" height="300" src="../img/post-transfer-tuning-inference-time.png"/>

### Evaulation

<img width="1000" height="300" src="../img/post-transfer-tuning-results.png"/>


<img width="1000" height="300" src="../img/post-transfer-tuning-edge-cpu.png"/>


<img width="500" height="200" src="../img/post-transfer-tuning-sequence-length.png"/>


<img width="1000" height="300" src="../img/post-transfer-tuning-server-class-cpu.png"/>



### Reference
[Transfer-Tuning: Reusing Auto-Schedules for Efficient Tensor Program Code Generation](https://arxiv.org/pdf/2201.05587)