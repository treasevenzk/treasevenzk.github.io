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
- ***One-Shot Tuner***: 这个还没有看，但是打算用这个代价模型融合到Heron里面
- ***Heron***: 目前在跑CPU部分的实验，本次重点了解代码中关于规则制定部分的内容，但是现在跑的实验还存在问题，就是我自己的电脑CPU和代码要求的不一致，我的电脑是不能用avx5只能用avx2,所以导致平台设置需要改变一下，另外代码要求的llvmshi 0.8版本，我的版本也不一致需要改一下

---

### 2.24-3.2进度
***论文阅读计划***
- ~~A Learned Performance Model For Tensor Processing Units~~
- ~~DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion~~
- ~~Optimal Kernel Orchestration for Tensor Programs with Korch~~
- ~~Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion~~
- ~~UNIT: Unifying tensorized instruction compilation~~ 
- ~~Lorien: Efficient Deep Learning Workloads Delivery~~
- ~~A Practical Tile Size Selection Model for Affine Loop Nests~~
- ~~AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations~~ 
- ~~Bring Your Own Codegen to Deep Learning Compiler~~
- ~~A Deep Learning Based Cost Model for Automatic Code Optimization~~

***论文复现工作***
- ***Heron***: 
重新梳理一下context类里面值的赋值问题，之前代码中存在如dense_i与P#ST:dense,AX:i互相乱用的问题，从而导致在生成schedule.py文件的时候发生字符串错乱的问题，目前该部分的代码都已经修改完整，目前是在CPU部分的实验，考虑GPU部分的实验也存在相应的问题在看GPU部分的代码需提前考虑一下；另外也了解和熟悉目前tvm存在的调度方法以及在该代码中采用的部分，目前在看部分的代码还是属于论文中的Generation部分，接下来开始看Exploration部分的代码，另外目前代码由于实验平台的限制导致不能采用tensorize调度，因为目前电脑的CPU支持avx2不支持avx512,tvm源码中关于tensorize部分的代码没有我电脑合适的张量化的方法，因此我在代码中是禁用tensorize，另外我生成的constraints.py文件中dense_global_tileAll = model.NewIntVar(1, 64, 'dense_global_tileAll')，与作者在网上提供的constraints.py有点不一致，考虑是这个原因导致后面会发生Illegal instruction (core dumped)的问题

- ***CMLCompiler***: 重新分析Hummingbird中关于决策树转化为矩阵的过程，分析目前CMLCompiler中没有关注到的点，关于Hummingbird中关于决策树转化部分的代码目前还没有看(下周可以具体分析)，目前在分析后面步骤的优化，另外在其实对这个想法的优化其实是集中对于GEMM算子的优化，另外他这个转换其实也是一个小的神经网络模型，我是不是应该考虑这个步骤分别对应于计算密集型算子、内存密集型算子，另外现在的想法没有结合决策树本身的特性，以及分析一下树的结构的工作下周继续开展一下，整理一下初步的小实验

***TVM源码学习***
整理学习TVM代码库实例讲解部分，目前初步对TVM整个代码结构有一定的了解，重点学习TVM中关于C++与Python之间如何相互调用的过程，学习PackedFunc的内容，关于下周可以结合Felix项目来配合TVM的源码学习

***这周状态***
总体而言还是比较不错，不过最近稍微有点焦虑，在思考自己的课题进度以及实验idea遇到一些困难，有点太敏感外界因素的影响，其实自己应该放下包袱，你没有什么好担心，专专心心地学习才是最重要的不要被外界的事情所干扰，沉下心来学习才是最重要的，另外不要太关注别人的进度，把握好自己的进度，每天的自己比昨天的自己有进步就是很棒的，不要拿自己去任何比较，这样既不会给自己带来任何的进步反而会影响自己的学习状态，赵凯加油，坚持下来，胜利是属于能坚持到最后的人的！！！好好规划好自己的每一天，让自己每一天都过得充实，而不是忙忙碌碌的一天

***学习计划***
每周每天还是要给自己制定相应的学习计划，坚持坚持坚持！！！

---

### 3.3-3.9进度
***论文阅读计划***
- ~~DISTAL: The Distributed Tensor Algebra Compiler~~ 
- ~~Optimizing the Memory Hierarchy by Compositing Automatic Transformations on Computations and Data~~
- Modeling the Interplay between Loop Tiling and Fusion in Optimizing Compilers Using Affine Relations 
- ~~MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators~~
- Fireiron: A Data-Movement-Aware Scheduling Language for GPUs 
- ~~DREW: Efficient Winograd CNN Inference with Deep Reuse~~ 
- ~~DeepCuts: A Deep Learning Optimization Framework for Versatile GPU Workloads~~ 
- ~~HASCO: Towards Agile HArdware and Software CO-design for Tensor Computation~~ 
- ~~FusionStitching: Boosting Memory Intensive Computations for Deep Learning Workloads~~ 
- ~~Atomic Dataflow based Graph-Level Workload Orchestration for Scalable DNN Accelerators~~ 
- ~~Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning~~
- ~~Tlp: A deep learning-based cost model for tensor program tuning~~

***论文复现工作***
- ***Heron***：
目前是跑通Heron/tests/quick_start/dlboost中的实验，验证之前关于Context、KnobManager、Solver类中的变量赋值问题，之前作者代码存在字符串混用的问题，


- ***Hummingbird***
dataset     iris
datadir     benchmark/operators/datasets/
modeldir    benchmarks/operators/models/
operator    all
backend     all(onnx-ml、hb-torchscript、hb-tvm)
cpus        1
batch_size  1000000
gpu         false
output      none
nrows       1000000
niters      5
validate    false
extra       {}



### 3.10-3.16进度
***论文阅读计划***
- ~~GTuner: Tuning DNN Computations on GPU via Graph Attention Network~~
- ~~Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators~~ 
- ~~Optimizing DNN Computation with Relaxed Graph Substitutions~~
- ~~AutoGraph: Optimizing DNN Computation Graph for Parallel GPU Kernel Execution~~
- ~~POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging~~
- ~~Collage: Seamless Integration of Deep Learning Backends with Automatic Placement~~ 
- Stateful dataflow multigraphs: a data-centric model for performance portability on heterogeneous architectures 
- ~~Exploiting Subgraph Similarities for Efficient Auto-tuning of Tensor Programs~~
- ~~Transferable Graph Optimizers for ML Compilers~~ 
- ~~Optimizing DNN computation graph using graph substitutions~~ 
- ~~IOS: Inter-Operator Scheduler for CNN Acceleration~~
- ~~Equality Saturation for Tensor Graph Superoptimization~~





AVX是Intel和AMD处理器上的SIMD指令集扩展
|指令集|AVX2|AVX512|
|"---"|"---"|"---"|
|寄存器宽度|使用256位宽的YMM寄存器|使用512位宽的ZMM寄存器，一次可处理更多数据|
|并行处理能力|同时处理8个32位浮点数或整数|同时处理16个32位浮点数或整数|
|指令集丰富度|比AVX512差|添加了许多新指令，包括更灵活的数据排列、更多的归约操作和更复杂的数学函数|
|处理器支持|在第四代Core及以后的Intel处理器和较新的AMD处理器上支持|在Intel Xeon Phi|
|功耗和频率影响|频率影响较小|导致CPU降频以保持在功耗和温度限制内|

读论文的技巧：
- 挑战：挖坑(新场景带来的新问题)、填坑(老场景老问题新思路)
- 方法：老方法解决新问题(看完场景和挑战就能猜到)、新的数据/硬件特征带来的新的方法组合、新的方法(少见、重点关注)
- 场景：如何对场景进行实验模拟、核心对比对象是谁，对比什么点、效果如何、那些东西没有比