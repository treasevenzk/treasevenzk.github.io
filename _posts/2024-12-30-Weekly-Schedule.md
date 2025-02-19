---
layout:     post
title:      Weekly Schedule
subtitle:   plan for every week
date:       2024-12-30
author:     Treaseven
header-img: img/bg32.png
catalog: true
tags:
    - Weekly Schedule
---


### 12.30-1.5进度
***论文阅读计划***
- ~~Interstellar: Using Halide’s Scheduling Language to Analyze DNN Accelerators~~
- ~~Analytical Characterization and Design Space Exploration for Optimization of CNNs~~
- ~~Mind mappings: enabling efficient algorithm-accelerator mapping space search~~
- ~~Optimizing Deep Learning Inference via Global Analysis and Tensor Expressions~~
- ~~AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-intensive ML Training and Inference on Modern SIMT Architectures~~
- ~~Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs~~


***论文复现工作***

- ***CMLCompiler***<br>
决策树算法sklearn转换成tvm的DL model的流程如下：<br>
cmlcompiler.model &rightarrow; build_model &rightarrow; init_model &rightarrow; __init__ &rightarrow; __parse_params__ &rightarrow; convert_decision_tree &rightarrow; __get_algo &rightarrow; decision_tree_classifier &rightarrow;

- ***Mind Mappings***<br>

本周进度还是相对较慢，因为中间元旦放假两天，耽误了一些进度，下周还是需要更加加油！！！

### 1.6-1.12进度
***论文阅读计划***
- ~~ROLLER: Fast and Efficient Tensor Compilation for Deep Learning~~
- ~~SmartMem: Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile~~
- ~~MonoNN: Enabling a New Monolithic Optimization Space for Neural Network Inference Tasks on Modern GPU-Centric Architectures~~
- ~~Accelerated Auto-Tuning of GPU Kernels for Tensor Computations~~
- ~~A full-stack search technique for domain optimized deep learning accelerators~~
- ~~RAMMER: Enabling Holistic Deep Learning Compiler Optimizations with rTasks~~
- ~~Bridging the Gap Between Domain-specific Frameworks and Multiple Hardware Devices~~ 

***论文复现工作***
- ***CMLCompiler***<br>
测试代码编写：测试的时候在用测试集，只需利用测试集中的一行数据进行测试，关注的是单次推理的性能，减少内存占用，~~之前的想法是用整个测试集进行测试~~
CMLCompiler的代码全部阅读完，但目前发现CMLCompiler在复现的时候效果不如论文中的结果，现在考虑重新写CMLCompiler的代码

- ***Heron***<br>
CPU的实验
config属性: out_name、method、max_trials、runner_number、runner_repeat、runner_timeout、build_timeout、in_dtype、out_dtype、cases、target_name、codegen_type、get_op、device_id、tuned

***这周状态***<br>
- 这周前四天效率还是比较高，但是后面三天效率比较低，下周还是要坚持6天效率比较高，加油

***每周学习的计划***<br>
- 这周没有制定相应的学习计划，但实际应该制定一些相应的学习计划，比如TVM源码的学习笔记，从下周开始必须整理相应的TVM源码的学习计划内容


### 1.13-1.19进度
***论文阅读计划***
- ~~DOPpler: Parallel Measurement Infrastructure for Auto-Tuning Deep Learning Tensor Programs~~
- ~~Mind the Gap: Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms~~
- ~~TIRAMISU: A Polyhedral Compiler for Expressing Fast and Portable Code~~ 
- ~~DISTAL: The Distributed Tensor Algebra Compiler~~
- ~~AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations~~
- ~~Tlp: A deep learning-based cost model for tensor program tuning~~
- ~~DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion~~


### 1.20-1.22进度
***论文阅读计划***
- ~~Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion~~ 
- ~~UNIT: Unifying tensorized instruction compilation~~ 
- ~~FreeTensor: A Free-Form DSL with Holistic Optimizations for Irregular Tensor Programs~~
- ~~Uncovering Nested Data Parallelism and Data Reuse in DNN Computation with FractalTensor~~
- ~~MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators~~
- ~~Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators~~
- ~~Apollo: Automatic Partition-based Operator Fusion through Layer by Layer Optimization~~


---

# 2025年每周工作进度表

### -2.16进度
***论文阅读计划***
- ~~A Comparison of End-to-End Decision Forest  Inference Pipelines~~
- ~~SilvanForge A Schedule Guided Retargetable Compiler  for Decision Tree Inference~~
- ~~Treebeard An Optimizing Compiler for Decision Tree Based ML Inference~~
- ~~Accelerating Decision-Tree-based Inference through Adaptive Parallelization~~
- ~~Tahoe Tree Structure-Aware High Performance Inference Engine for Decision Tree Ensemble on GPU~~

***论文复现工作***

- ***Heron***: ~~目前将该代码中的CPU部分的实验过程全部梳理了一遍，但对调度规则制定和调度搜索这一块的代码还没有具体详细看，接下来继续整理这一块的内容，然后在比较GPU部分的实验~~
- ***Hummingbird***: ~~代码中关于tvm部分的实验梳理一遍，还没有准确定位关于决策树三种算法的选择块的代码，需要跑通这一块GPU、CPU部分的实验，作为后续对比实验的数据~~,目前发现我在做实验过程利用hummingbird这个代码对决策树进行推理的效果不如sklearn好，使用hummingbird的版本为0.4.12，tvm的版本为0.15.0，然后运行hummingbird在tvm上会报下面的错误，估计这个错跟tvm的版本有关，在构建计算图的时候发生了一些错误导致的，可以考虑tvm版本的事情，选择一个稳定的tvm版本来做实验，另外论文中使用的tvm版本是0.6.0，这里建议和github仓库维护者反应一下这个问题
```
[15:51:03] /home/newuser/tvm/src/te/schedule/bound.cc:119: not in feed graph consumer = compute(p0_red_temp, body=[T.reduce(T.comm_reducer(lambda argmax_lhs_0, argmax_lhs_1, argmax_rhs_0, argmax_rhs_1: (T.Select(argmax_lhs_1 > argmax_rhs_1 or argmax_lhs_1 == argmax_rhs_1 and argmax_lhs_0 < argmax_rhs_0, argmax_lhs_0, argmax_rhs_0), T.Select(argmax_lhs_1 > argmax_rhs_1, argmax_lhs_1, argmax_rhs_1)), [-1, T.float32(-3.4028234663852886e+38)]), source=[k1, p0[ax0, k1]], init=[], axis=[T.iter_var(k1, T.Range(0, 251), "CommReduce", "")], condition=T.bool(True), value_index=0), T.reduce(T.comm_reducer(lambda argmax_lhs_0, argmax_lhs_1, argmax_rhs_0, argmax_rhs_1: (T.Select(argmax_lhs_1 > argmax_rhs_1 or argmax_lhs_1 == argmax_rhs_1 and argmax_lhs_0 < argmax_rhs_0, argmax_lhs_0, argmax_rhs_0), T.Select(argmax_lhs_1 > argmax_rhs_1, argmax_lhs_1, argmax_rhs_1)), [-1, T.float32(-3.4028234663852886e+38)]), source=[k1, p0[ax0, k1]], init=[], axis=[T.iter_var(k1, T.Range(0, 251), "CommReduce", "")], condition=T.bool(True), value_index=1)], axis=[T.iter_var(ax0, T.Range(0, 103069), "DataPar", "")], reduce_axis=[T.iter_var(k1, T.Range(0, 251), "CommReduce", "")], tag=comm_reduce_idx, attrs={})
```
- ***CMLCompiler***: 1. ~~发现中关于决策树转换部分的代码还存在问题，同时需要决策树变换结果打印出来，仔细分析一下是否还可以采用调度方法来提高，目前打算这个代码作为一个baseline然后进行修改;~~ 2.决策树训练的问题，之前是使用yearPredicition数据集里面的全部特征，后面和作者联系发现作者在训练决策树模型只使用了27个特征，同时也约束了决策树的深度，解决一开始直接用决策树去训练得到的一个模型然后在用cmlcompiler会出现内存溢出的问题；3.关于上面hummingbird出现的错在cmlcompiler也会出现，然后发现确实是在构建计算图的时候出错，然后我对tree_common.py文件中tree_gemm这个函数进行修改，出错的原因是relay.argmax的规约操作导致在TVM的调度时出现边界推断问题，从而导致编译失败，修改的操作将argmax分解为更基本的操作，明确指定了所有中间步骤的形状和类型


*** TVM源码学习 ***

目前在看TVM官网上的开发教程，这周还没有怎么看，下周我一定要好好看


### 2.17-2.23进度
***论文阅读计划***
- ~~One-Shot Tuner for Deep Learning Compilers~~
- ~~A Flexible Approach to Autotuning Multi-Pass Machine Learning Compilers~~
- ~~Transfer-Tuning: Reusing Auto-Schedules for Efficient Tensor Program Code Generation~~
- ~~Effective Performance Modeling and Domain-Specific Compiler Optimization of CNNs for GPUs~~
- ~~Alcop: Automatic load-compute pipelining in deep learning compiler for ai-gpus~~
- ~~Automatic Generation of Multi-Objective Polyhedral Compiler Transformations~~
- ~~Welder: Scheduling Deep Learning Memory Access via Tile-graph~~
- ~~Bolt: Bridging the gap between auto-tuners and hardware-native performance~~
- ~~A Holistic Functionalization Approach to Optimizing Imperative Tensor Programs in Deep Learning~~
- ~~Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion~~
- ~~Apollo: Automatic Partition-based Operator Fusion through Layer by Layer Optimization~~
- ~~Fasor: A Fast Tensor Program Optimization Framework for Efficient DNN Deployment~~


***论文复现工作***
- ***One-Shot Tuner***: 



### 2.24-3.2进度
***论文阅读计划***
- A Learned Performance Model For Tensor Processing Units
- DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion
- Optimal Kernel Orchestration for Tensor Programs with Korch
- Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion
- UNIT: Unifying tensorized instruction compilation 
- Lorien: Efficient Deep Learning Workloads Delivery
- A Practical Tile Size Selection Model for Affine Loop Nests
- AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations 
- Bring Your Own Codegen to Deep Learning Compiler
- A Deep Learning Based Cost Model for Automatic Code Optimization