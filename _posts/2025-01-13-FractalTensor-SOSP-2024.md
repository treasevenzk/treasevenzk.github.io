---
layout:     post
title:      FractalTensor SOSP 2024
subtitle:   Uncovering Nested Data Parallelism and Data Reuse in DNN Computation with FractalTensor
date:       2025-01-13
author:     Treaseven
header-img: img/bg18.jpg
catalog: true
tags:
    - Deep Learning Compiler
    - Loop Program Analysis
    - Nested Data Parallelism
---


### Existing Method's Problems
- DAG is less expressive and problematic to support many DNN algorithms
- users either use a more flexible, imperative programming interface like pytorch to implement new DNNs while sacrificing efficiency, or keep introducing new tensor operators with optimized performance but ad-hoc semantics based on developer's experience


<img width="500" height="500" src="../img/post-fractaltensor-limitions.png"/>

***Challenges***
- to identify and exploit the obscure data parallelism, especially in the presence of complex, fine-grained data dependencies across operator boundaries or across nested loops
- to identify and exploit the subtle data reuse opportunities across operators or nested loops

***Opportunities***
- the diverse DNN computation patterns can be expressed by a combination of second-order array compute operators
- the high-level data access patterns during DNN computation are highly stylized and can be expressed by a few first-order array access operators

### Programming FractalTensor

<img width="1000" height="500" src="../img/post-fractaltensor-overview.png"/>

#### Extended task dependence graph
- Operation node
- Buffer node
- Block node
- Access map



### System Implementation


### Evaluation

<img width="1000" height="450" src="../img/post-fractaltensor-end-to-end-performance.png"/>

<img width="1000" height="400" src="../img/post-fractaltensor-rnn-performance.png"/>

<img width="500" height="300" src="../img/post-fractaltensor-memory.png"/>