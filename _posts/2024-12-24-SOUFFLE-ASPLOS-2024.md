---
layout:     post
title:      SOUFFLE ASPLOS 2021
subtitle:   Optimizing Deep Learning Inference via Global  Analysis and Tensor Expressions
date:       2024-12-24
author:     Treaseven
header-img: img/bg32.png
catalog: true
tags:
    - Deep Neural Network
    - Compiler Optimization
    - Tensor Expression
    - GPU
---

### Contributions
- a tensor-expression-based global analysis to identify critical partitioning points
- a semantic preserving transformations approach that use affine transformation to simplify the tensor expressions of each subprogram


### Motivation


<img width="1000" height="300" src="/img/post-souffle-working-example.png"/>

***Fail to explore optimization between memory- and compute-intensive kernels***: manually crafted rules cannot cover a diverse set of computation patterns and miss the optimization opportunity in this case<br>
***Suboptimal fusion strategy for reduction operators***<br>
***Poor optimiztions across computation-intensive kernels***

post-souffle-example.png

<img width="500" height="900" src="/img/post-souffle-example.png"/>


### Global Computation Graph Analysis

- identifying data reuse opportunities
- intra-TE element-wise dependency analysis
- TE characterization
- TE Program Partitioning


### Semantic-preserving TE Transfromations

- Horizontal transformation for independent TEs

<img width="500" height="200" src="/img/post-souffle-horizontal.png"/>

- Vertical transformation for one-relies-on-one TEs

<img width="500" height="200" src="/img/post-souffle-vertical.png"/>

- Schedule TEs

- Merging TEs Schedule

- Optimizations within a Subprogram: Instruction-level optimization、Tensor reuse optimization

- Put it all together

<img width="500" height="700" src="/img/post-souffle-algorithm.png"/>

### Evaluation



<img width="500" height="200" src="/img/post-souffle-end-to-end.png"/>



<img width="500" height="200" src="/img/post-souffle-execution.png"/>



<img width="500" height="200" src="/img/post-souffle-gpu.png"/>



<img width="500" height="200" src="/img/post-souffle-breakdown.png"/>