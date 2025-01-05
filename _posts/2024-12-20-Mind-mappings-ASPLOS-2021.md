---
layout:     post
title:      Mind mappings ASPLOS 2021
subtitle:   Mind Mappings Enabling Efficient Algorithm-Accelerator Mapping Space Search
date:       2024-12-20
author:     Treaseven
header-img: img/bg32.png
catalog: true
tags:
    - Programmable Domain-specific Acclerators
    - Mapping Space Search
    - Gradient-based Search
---

### Background
- algorithm-accelerator mapping space
- mapping space search
- cost function


### Method

<img width="1000" height="400" src="../img/post-mind-mapping-procedure.png"/>

#### Phase 1: Approximating the Map Search Space
***Generating the surrogate model training set***: <br>
which map spaces should be used to populate the training set?<br>
based on the choice of map spaces, which mappings should we sample to populate the training set<br>
how to uniquely associate each mapping m with its map space $M_{a, p}$?<br>
how to calculate cost per mapping?<br>

***Input Mapping Representation***

***Output Cost Representation***

#### Phase 2: Gradient Search to Find High-Quality Mappings


### Evaluation

post-mind-mappings-iso-iteration.png

post-mind-mappings-iso-time.png

post-mind-mappings-experiments.png