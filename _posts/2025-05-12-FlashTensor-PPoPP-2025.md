---
layout:     post
title:      FlashTensor PPoPP 2025
subtitle:   FlashTensor Optimizing Tensor Programs by  Leveraging Fine-grained Tensor Property
date:       2025-05-12
author:     Treaseven
header-img: img/bg2.jpg
catalog: true
tags:
    - Deep learning compiler
    - Long context
    - Memory optimization
---

### Motivation
在长文本场景里面会产生极度大的中间变量会导致大量内存开销


### System Overview

<img width="500" height="400" src="../img/post-flashtensor.png"/>

**Tensor Property Identifier**
Property Definition
reduce dependency： NonPara、Reuse、Batch
broadcast
size
value

Dataflow-Based Property Identification
Property Propagation
Property Aggregation

**Tensor Property-Aware Optimization**
