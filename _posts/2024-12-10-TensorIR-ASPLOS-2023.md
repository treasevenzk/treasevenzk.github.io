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
* 识别张量计算机会
* 构建程序框架
* 优化具体参数
<img width="1000" height="600" src="/img/post-tensorir-automate.png"/>
引入AutoCopy块：将数据移动作为独立的优化目标、支持灵活的数据布局转换和缓存策略
端到端的优化流程：从高层程序到硬件指令的完整映射；自动处理计算和数据移动的优化


#### Tensorization Candidate Generation
<img width="1000" height="400" src="/img/post-tensorir-candidate.png"/>
ReIndex的目的是简化访问模式，通过引入中间缓冲区Ar和Br，将复杂的索引计算分解为两步

```
# 重组输入A的访问模式
Ar[n, h, w, rh, rw, rc] = A[n, h*2+rh, w*w+rw, rc]
# 重组卷积核B的访问
Br[c, rh, rw, rc] = B[c, rh, rw, rc]
# 最终计算
Cr[n, h, w, c] += Ar[n, h, w, rh, rw, rc] * Br[c, rh, rw, rc]
```

将卷积运算映射到矩阵乘法硬件指令
```
硬件指令的形式
C[x, y] += A[x, k] * B[k, y]
特征向量χ(v)表示一个迭代器v在不同缓冲区中的出现情况：对每个迭代器，生成一个二进制向量[是否在C中，是否在A中，是否在B中]；比如χ(n)=[1,1,0]表示n出现在C和A中，但不在B中
n, h, w: χ(n) = χ(h) = χ(w) = [1,1,0]
c:       χ(c) = [1,0,1]
rh, rw, rc: χ(rh) = χ(rw) = χ(rc) = [0, 1, 1]
与目标硬件指令对应
x: χ(x) = [1, 1, 0] # 对应n, h, w的模式
y: χ(y) = [1, 0, 1] # 对应c的模式
k: χ(k) = [0, 1, 1] # 对应rh,rw,rc的模式
```
根据上面的映射将n,h,w三个维度合并成一个维度、rh,rw,rc三个维度合并成一个维度，然后重写成如下操作
```
Ct[fuse(n,h,w), c] += At[fuse(n,h,w), fuse(rh,rw,rc)] * Bt[fuse(rh,rw,rc), c]
```

#### Tensorized Program Sketch Generation
Computation Schedule & Data Movement
传统优化中，计算和数据移动的时间差距较小，使用张量运算原语后，差距显著增大；如果不专门优化数据移动，很容易出现计算单元空闲等待数据的情况
```
# 原始代码
for i in range(64):
    for j in range(64):
        C[i, j] = A[i, j] + B[i, j]

# 计算调度优化
for i_outer in range(0, 64, 16):
    for j_outer in range(0, 64, 16):
        for i_inner in range(16):
            for j_inner in range(16):
                i = i_outer + i_inner
                j = j_outer + j_inner
                C[i, j] = A[i, j] + B[i, j]

# 数据移动优化
for i_outer in range(0, 64, 16):
    for j_outer in range(0, 64, 16):
        # 数据移动：加载到共享内存
        A_shared = load_to_shared_memory(A[i_outer:i_outer+16, j_outer:j_outer+16])
        B_shared = load_to_shared_memory(B[i_outer:i_outer+16, j_outer:j_outer+16])

        # 计算使用共享内存中的数据
        for i_inner in range(16):
            for j_inner in range(16):
                i = i_outer + i_inner
                j = j_outer + j_inner
                C[i, j] = A_shared[i_inner, j_inner] + B_shared[i_innner, j_inner]
```

### Evaluation
#### Single Operator Evaluation


<img width="500" height="500" src="/img/post-tensorir-single.png"/>


<img width="500" height="500" src="/img/post-tensorir-single-platform.png"/>


#### End-to-End Model Evaluation
<img width="500" height="500" src="/img/post-tensorir-end-performance.png"/>


#### ARM CPU Evaluation
<img width="500" height="400" src="/img/post-tensorir-arm-single.png"/>



<img width="500" height="300" src="/img/post-tensorir-arm-end.png"/>


### Thinking
(1) ***自动调度器的限制***
(2) ***tensorization候选生成的局限***
(3) ***数据移动优化***

### Reference
[TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://dl.acm.org/doi/pdf/10.1145/3575693.3576933)