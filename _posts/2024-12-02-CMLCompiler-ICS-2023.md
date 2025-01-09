---
layout:     post
title:      CMLCompiler ICS 2023
subtitle:   CMLCompiler A Unified Compiler for Classical Machine Learning
date:       2024-12-02
author:     Treaseven
header-img: img/bg24.jpg
catalog: true
tags:
    - Classical Machine Learning
    - Deep Learning
    - Compiler
---

### Motivatioin
leverage DL's well-defined unified abstractions and highly mature compilers, optimization technologies, and frameworks<br>
面临的挑战：<br>
(1) 深度学习算子关注张量，经典机器学习关注数组、矩阵、标量和表格<br>
(2) 深度学习模型都是神经网络模型，经典机器学习模型几乎不能用神经网络表示

### The Unified Abstraction
#### Operator Representation

<img width="1000" height="400" src="/img/post-cmlcompiler-example.png"/>

***The features of CML operator representations***: 1. the weights of DL operators and CML operator representations have different meanings 2. the frequent operators in DL and CML are not the same

#### Extended Computational Graph
设计的动机：传统DL计算图默认处理密集的float32类型数据，CML模型中很多操作和权重天然是稀疏或低精度的，直接使用DL计算图会忽略这些特性带来的优化机会

### Design

<img width="500" height="600" src="/img/post-cmlcompiler.png"/>

#### Graph Optimizer


<img width="1000" height="300" src="/img/post-cmlcompiler-graph-rewriting.png"/>

***Dtype rewriting***: 使用低精度计算替代高精度计算


<img width="500" height="400" src="/img/post-cmlcompiler-dtype-rewriting.png"/>

***Sparse operator replacing***：识别稀疏权重，将密集算子替换为稀疏实现<br>
***Redundant elimination***：消除不影响最终结果的冗余算子

### Evaluation


<img width="1000" height="300" src="/img/post-cmlcompiler-performance.png"/>



<img width="1000" height="400" src="/img/post-cmlcompiler-execution.png"/>



<img width="1000" height="400" src="/img/post-cmlcompiler-latency.png"/>

### Thinking
(1) 支持算子有限，如矩阵分解算法(SVD相关:矩阵正交化算子、特征值计算算子、特征向量计算算子；PCA相关：协方差矩阵计算算子、降维映射算子；NMF相关)、概率模型(贝叶斯模型：条件概率计算算子、后验概率更新算子、边缘化算子；EM算法：期望计算算子、最大化算子)；降维和流形学习、时间序列算法、集成学习(Stacking: 模型组合算子、预测融合算子；Blending：权重计算算子、加权平均算子、预测集成算子)<br>
(2) 优化方法局限：主要依赖三种图重写优化(算子级优化：算子融合优化、内存优化、向量化优化；图级优化)、缺乏自动优化策略的研究(自动调优)、跨层优化怎么做，这个点在深度模型中目前也没有做到<br>
(3) 评估方面的不足：基准测试数据集单一(主要使用YearPrediction数据集、缺乏在不同规模和特征的数据集上的评估、可以使用更多真实场景的数据集测试)

||CMLCompiler|Hummingbird|
|:---|:---:|:---:|
|相同点|将CML转换为深度学习表示的方式、使用张量计算作为底层计算范式|
|抽象层次|算子表示和扩展计算图(ECG)、直接面向底层编译优化|基于ONNX中间表示，更关注高层模型转换|
|优化策略|提供专门的图重写优化|依赖DL框架的通用优化|
|实现方式|基于TVM实现完整的编译框架，直接生成优化的计算图|通过PyTorch/ONNX等框架转换，间接使用DL框架的优化能力|


### 补充材料
Binaraizer:用于将数值特征转换为二值
MinMaxScaler: 用于将特征缩放到指定范围
MaxAbsScaler: 通过除以每个特征的最大绝对值来进行缩放
StandScaler: 标准化缩放
RobustScaler: 稳健缩放



### Reference 
[CMLCompiler: A Unified Compiler for Classical Machine Learning](https://dl.acm.org/doi/pdf/10.1145/3577193.3593710)