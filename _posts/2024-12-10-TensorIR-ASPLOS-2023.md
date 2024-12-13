---
layout:     post
title:      TensorIR ASPLOS 2023
subtitle:   TensorIR An Abstraction for Automatic Tensorized Program Optimization
date:       2024-12-10
author:     Treaseven
header-img: img/bg7.jpg
catalog: true
tags:
    - Machine Learning Compiler
    - Deep Neural Network
    - Tensor Computation
--- 

### Motivation
* 现代硬件加速器引入专门的张量计算原语
* 传统手动优化库开发成本高，难以适应快速变化的模型和硬件
* 需要自动化编译方法来利用这些硬件加速能力
面临的挑战
(1) ***Abstraction for Tensorized Programs***：需要一个能表达等价张量化计算的抽象
(2) ***Large Design Space of Possible Tensorized Program Optimizations***：需要在大规模设计空间中找到优化方案
提出TensorIR
(1) 引入block构造来分离张量计算
(2) 构建具有正确性验证的变换原语
(3) 设计新的张量化感知自动调度器

### Overview

<img width="1000" height="500" src="/img/post-tensorir-overview.png"/>

### TensorIR Abstraction
#### Block
<img width="500" height="600" src="/img/post-tensorir-block.png"/>

#### Scheduling Transformations
可调度性的定义<br>
(1) 一个block是"可调度的".如果它只包含以子block作为叶节点的循环嵌套<br>
(2) 通过分析子block的签名核依赖信息来转换这些可调度block中的循环嵌套<br>
(3) 可调度block可以包含不可调度的子block

***schedule primitive***
* Loop Transformations
* Blockization
* Separation of Scheduling and TensorIR

<img width="500" height="600" src="/img/post-tensorir-scheduling-loop.png"/>


<img width="500" height="400" src="/img/post-tensorir-scheduling-block.png"/>


#### Validation

***Loop Nesta Validation***: 验证迭代器绑定是否满足迭代域约束<br>
***Threading Validation***: 线程绑定、协作内存访问、执行作用域<br>
***Correctness of Schedule Primitives***: 当原语只改变循环嵌套时可以用于验证过程确保正确性；对于blocks的原语找到原语特定的必要条件


### Auto-Scheduling Tensorized Programs
<img width="1000" height="600" src="/img/post-tensorir-automate.png"/>

#### Tensorization Candidate Generation
<img width="1000" height="400" src="/img/post-tensorir-candidate.png"/>

### Evaluation
#### Single Operator Evaluation


<img width="500" height="500" src="/img/post-tensorir-single.png"/>


<img width="500" height="500" src="/img/post-tensorir-single-platform.png"/>


#### End-to-End Model Evaluation
<img width="500" height="500" src="/img/post-tensorir-end-performance.png"/>


#### ARM CPU Evaluation
<img width="500" height="400" src="/img/post-tensorir-arm-single.png"/>



<img width="500" height="300" src="/img/post-tensorir-arm-end.png"/>

### Reference
[TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://dl.acm.org/doi/pdf/10.1145/3575693.3576933)