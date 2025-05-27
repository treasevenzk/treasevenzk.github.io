---
layout:     post
title:      HeronCode
subtitle:   Code Reproduction
date:       2025-01-12
author:     Treaseven
header-img: img/bg20.jpg
catalog: true
tags:
    - debug
---
TVM中的内容:
from tvm.autotvm.measure.measure import MeasureInput: MeasureInput类在TVM的AutoTVM模块中的作用是封装测量特定张量操作配置性能所需的信息；存储任务(要优化的张量操作)和要测量的特定配置、包含测量基础设施编译和运行操作特定实现所需的信息、作为输入提供给实际基准测试不同配置性能的测量模块；有助于为特定硬件目标找到张量操作的最佳实现方案

论文中要求的规则：
Rule name                     Condition and Application
Always-Inline                 IsStrictInlinable(S, i)
                              S' ← Inline(S, i); i' ← i-1
Multi-level-Tiling            HasDataReuse(S, i)
                              S' ← MultiLevelTiling(S, i); i' ← i-1
Add-Cache-Stage               HasDataReuse(S, i)&
                              ¬ HasFusibleConsumer(S, i)
                              S' ← AddCacheWrite(S, i); i' ← i-1
User-Defined-Rule             user-defined transformations and conditions
Tensorize                     Tensoriable(S, i)
                              S' ← Tensorize(S, i); i' ← i
Add Multi-level SPM           HasDataReuse(S, i) &
                              HasMultiLevelCache(S, i)
                              S' ← AddMultiLevelCacheRead(S, i); i' ← i
Add Multi-Scope SPM           HasDataReuse(S) &
                              HasMultiScopeCache(S, i)
                              S' ← AddCacheReadInputs(S, i);
                              S'' ← AddCacheWrite(S', i); i' ← i-1


调度原语的作用：
split: 将一个大的循环分解成内外两层循环，便于后续的并行化和向量化优化，通过选择split的factor大小，使内层循环的数据块大小与CPU的L1、L2缓存大小匹配，避免缓存颠簸(cache thrashing)，提高局部性，有利于硬件预取机制发挥作用，提前将即将需要的数据加载到缓存中
reorder: 通过重排循环顺序，使内存访问模式更符合硬件特性，减少缓存miss和内部带宽压力，让访问连续内存的循环放在内层循环，提高空间局部性
parallel: 将指定的循环并行化，将循环的不同迭代分配给不同的线程或处理器核心同时执行，注意点： 只有迭代间没有依赖关系的循环才能安全地并行化，并行化会引入线程创建和同步开销，对于计算量较小的循环可能得不偿失，并行度不应超过可用的物理核心数太多，避免上下文切换开销
fuse: 将多个嵌套循环合并成一个循环，降低循环控制开销，改善并行粒度，增强cache局部性，简化调度复杂性，适应硬件特征
bind: 将循环轴绑定到GPU的线程层次结构，明确指定循环的不同维度如何映射到GPU的线程层次结构
compute_at: 控制计算的位置和顺序，实现计算局部性优化，将一个计算操作安排在另一个计算的特定循环级别执行，改变默认的“计算全部-然后使用”的模型，数据局部性优化通过将生产者操作移动消费者的循环中，减少中间结果的存储需求，提高缓存利用率，减少生产者和消费者之间的延迟
cache_read: 创建数据的本地缓存副本，为输入张量创建一个临时缓冲区，用于缓存数据，通过本地缓存减少对主内存或全局内存的反复访问，降低内存带宽压力
cache_write: 为计算结果创建临时缓冲区，然后再写入最终目标位置，为计算结果创建一个中间缓冲区，允许在不同内存层次间优化写入操作，避免多个线程直接写入同一个全局内存，减少内存争用和冲突
unroll: 展开循环，增加指令级并行，减少分支预测失误，该技术需要权衡，过度展开会增加指令缓存压力和代码大小，而展开得太少则无法充分利用优化机会，通过对小的固定大小循环应用unroll最为有效
pragma: 用于向代码生成器传递特定的编译提示或指令，控制生成代码的特性或触发特定的硬件功能，如特殊指令集、硬件预取、向量化、循环边界检查
stroage_inline: 用于控制数据在内存中的对齐方式
vectorize: 用于将循环转换为向量操作，利用SIMD，根据目标平台，可能会映射到具体的SIMD指令集，x86平台上可能使用SSE、AVX或AVX-512指令，ARM平台上可能使用NEON指令，CUDA平台上可能映射到向量化内存访问
tensorize: 用于将计算模式映射到硬件加速指令或专用计算单元，充分利用深度学习加速器和特殊硬件功能，将常见的计算模式直接映射到硬件提供的特殊指令或计算单元上，如NVIDIA的Tensor Core、ARM的矩阵乘法引擎等，针对深度学习加速器提供的特殊计算单元进行优化，直接使用它们的原生操作
compute_inline: 用于将计算操作内联到它的消费者，消除中间缓冲区，合并计算操作， 与compute_at形成互补，compute_at控制计算在何处发生，而compute_inline则完全融合计算

Heron调度模板生成的逻辑
1. 首先检查并内联注入式算子(ctx.InlineAll(s))
2. 根据平台特性(如Tensorcore)生成特定的内存层次结构(organize)
3. 对更新后的计算图中的每个阶段，根据规则应用相应的调度

设计上面顺序的原因： 内联→内存层次→其他优化
1. 内联注入式算子的优先性：简化后续优化(内联后的计算图更加简洁，减少需要考虑的计算阶段)、避免冗余优化、便于识别计算模式(内联后更容易识别出张量化的计算模式)
2. 内存层次结构的确定：硬件约束的核心部分(内存层次直接映射到硬件架构：如全局内存→共享内存→寄存器)、数据移动成本主导(在DLA上，数据移动通常比计算更耗时，因此内存层次优先确定)、影响后续所有优化(循环分块、循环顺序等都依赖于已确定的内存层次结构)
3. 其他调度决策的依赖性: 循环分块(分块大小需要基于已确定的内存层次结构)、向量化/展开(依赖于内存访问模式，而内存访问模式又由内存层次决定)、计算位置(计算应该在哪一层内存执行，取决于内存层次结构)

TensorCore计算流程：
TensorCore的计算流程基于特殊的矩阵乘法指令(如mma.sync),这些指令要求:
1.输入矩阵必须加载到特定的WMMA片段中
2.结果累积到WMMA累加器中
3.结构从WMMA累加器存储回普通内存

addCacheTensorCoreOp代码对应的内存层次的建立：
原始输入→共享内存→WMMA输入片段→WMMA计算→WMMA累加器→共享内存→全局内存

cache_write和cache_read在代码中的实际用途：
cache_write用于累积器，虽然cache_write是为计算结果创建临时缓冲区，在代码中，它创建WMMA累加器空间，作为矩阵乘法的目标，"wmma.accumulator"是特殊的存储范围，映射到TensorCore硬件的累加器
cache_read用于多种目的，从全局内存到共享内存的加载，从共享内存到WMMA矩阵片段的加载，从WMMA累加器结果到共享内存的读取(这看起来像是存储，但在TVM中是通过cache_read实现)


块级别：通常是较外层的循环，处理数据的较大分块
线程级别：较内层的循环，通常对应于单个线程的工作
更内层循环：在线程内部的循环，通常对应于单个线程内部的操作
在TensorCore编程中，循环层次对应的典型计算位置：
块级别循环→共享内存操作
线程级别循环→线程私有操作
内层循环→WMMA指令单元操作

Heron调度应用的决策逻辑
**先通用后特化**：系统首先尝试应用通用调度策略(如defaultSharedLoadSched和defaultGPUSched),如果这些不适用，才会尝试组合多个特定调度策略
共享内存加载且没有可融合的消费者时使用defaultSharedLoadSched的原因：1.内存加载优化的特殊需求 2.无可融合消费者的情况：这个加载操作需要作为独立阶段执行，加载的数据需要完全写入共享内存后才能被后续操作使用，在加载和计算之间存在同步点

TCStartOP
当前阶段在ctx.compute_poses说明是需要在其他阶段内计算的阶段，执行compute_at操作，将其放置在指定位置，更新轴长度信息
更新轴长度信息的原因：1.自动推导约束 2.生成更准确的约束条件：内存容量约束(共享内存大小限制)、计算单元约束(WMMA矩阵大小要求)、线程组织约束(线程块大小限制)

defaultSharedLoadSched
为共享内存加载阶段设置默认调度策略：
1.存储对齐：确保共享内存访问是对齐的，减少bank冲突
2.循环融合：将多个循环融合，简化后续分割
3.向量化：利用向量直径加速内存加载
4.线程绑定：将循环并行化到GPU线程

hasFuisbleConsumer明确哪个阶段负责调度共享内存操作，避免冗余优化(不对已有明确消费者的阶段应用默认优化)
在TensorCore编程中，共享内存加载大致有两种模式：
模式A：直接由消费者内联(共享内存阶段与其消费者紧密耦合，消费者可以融合共享内存加载到自己的循环中、消费者负责对共享内存加载进行调度)
模式B：独立调度(共享内存阶段相对独立、需要单独的调度策略、需要明确的内存对齐、向量化和线程绑定)
判断算子是否与其消费者紧密耦合，依据是该算子的消费者只有一个，或者算子的消费者的生产者只有该算子

defaultSchedOp：基于标签的调度选择机制提供一种灵活、可扩展的方式来为不同类型的计算操作应用不同的调度策略
基于GPU架构特性和并行计算模型来设计通用GPU调度策略，能够有效利用GPU的并行计算能力
调度顺序：融合(fuse)→向量化(vectorize)→线程绑定(thread binding)→块绑定(block binding) 创建一个层次化的并行执行模型，从最细粒度(向量指令)到最粗粒度(块)


融合消费者是指能够与当前操作进行循环融合优化的下游操作
当一个操作有"可融合的消费者"，意味着：1.消费关系明确，该操作的输出直接且唯一地被另一个操作使用 2. 一对一关系，该操作是其消费者的唯一输入来源 3.访问模式匹配，两个操作的循环迭代空间可以对齐和合并
当一个操作没有"可融合的消费者"时，可能是因为：1.多个消费者，该操作的输出被多个下游操作使用 2.消费者有多个输入，下游操作不仅使用该操作的输出，还使用其他操作的输出 3.无消费者，该操作是最终输出，没有下游操作

tsbo代码经历过程

becbo代码经历过程
fun CantileverBeamDesign
algo BECBO
get_initial_dataset
algo.fit_surrogate
algo.optimize: self.get_acq -> get_constr_lb_dynamic -> get_feasible_prob -> find_boundary_sample


数据清洗过程
operator        参数量          参数
bmm             45             ['wmma_m', 'wmma_k', 'wmma_n', 

'batch_matmul.wmma.accumulator_shared_pos', 'batch_matmul.wmma.accumulator_local_pos', 'batch_matmul_shared_pos', 'batch_matmul.wmma.accumulator.shared_local_pos', 'batch_matmul_unroll_pragma',

 'batch_matmulb.innertileSpatial', 'batch_matmuli.innertileSpatial', 'batch_matmulj.innertileSpatial', 

 'batch_matmulb.inner.innertileSpatial', 'batch_matmuli.inner.innertileSpatial', 'batch_matmulj.inner.innertileSpatial', 'threadIdx.y', 'batch_matmulb.inner.inner.innertileSpatial', 'batch_matmuli.inner.inner.innertileSpatial', 'batch_matmulj.inner.inner.innertileSpatial', 'threadIdx.x', 'batch_matmulb.inner.inner.inner.innertileSpatial', 'batch_matmuli.inner.inner.inner.innertileSpatial', 'batch_matmulj.inner.inner.inner.innertileSpatial', 'batch_matmul.wmma.accumulator.shared_ax2', 'batch_matmul.wmma.accumulator.shared_offset', 'batch_matmul.wmma.accumulator.sharedax0tileSpatial', 'batch_matmul.wmma.accumulator.sharedax1tileSpatial', 'batch_matmul.wmma.accumulator.sharedax2tileSpatial', 'batch_matmul.wmma.accumulatorb.ctileAll', 'batch_matmul.wmma.accumulatori.ctileAll', 'batch_matmul.wmma.accumulatorj.ctileAll', 'batch_matmul.wmma.accumulatorktileAll', 'batch_matmul.wmma.accumulatorb.c.innertileAll', 'batch_matmul.wmma.accumulatori.c.innertileAll', 'batch_matmul.wmma.accumulatorj.c.innertileAll', 'batch_matmul.wmma.accumulatork.innertileAll', 'batch_matmul.wmma.accumulatorb.c.inner.innertileAll', 'batch_matmul.wmma.accumulatori.c.inner.innertileAll', 'batch_matmul.wmma.accumulatorj.c.inner.innertileAll', 'batch_matmul.wmma.accumulatork.inner.innertileAll', 'B.shared_ax2', 'B.shared_offset', 'B.shared_vectorize', 'A.shared_ax2', 'A.shared_offset', 'A.shared_vectorize']


['wmma_m', 'wmma_k', 'wmma_n', 

'batch_matmul.wmma.accumulator_shared_pos', 'batch_matmul_wmma_accumulator_local_pos', 'batch_matmul_shared_pos', 'batch_matmul_wmma_accumulator_shared_local_pos', 'batch_matmul_unroll_pragma',

 'batch_matmulb_innertileSpatial', 'batch_matmuli_innertileSpatial', 'batch_matmulj_innertileSpatial', 

 'batch_matmulb_inner_innertileSpatial', 'batch_matmuli_inner_innertileSpatial', 'batch_matmulj_inner_innertileSpatial', 
 'threadIdx_y', 

 'batch_matmulb_inner_inner_innertileSpatial', 'batch_matmuli.inner.inner.innertileSpatial', 'batch_matmulj.inner.inner.innertileSpatial', 

 'threadIdx_x', 
 'batch_matmulb_inner_inner_inner_innertileSpatial', 'batch_matmuli_inner_inner_inner_innertileSpatial', 'batch_matmulj_inner_inner_inner_innertileSpatial', 
 'batch_matmul_wmma_accumulator_shared_ax2', 'batch_matmul.wmma.accumulator.shared_offset', 
 'batch_matmul.wmma.accumulator.sharedax0tileSpatial', 'batch_matmul.wmma.accumulator.sharedax1tileSpatial', 'batch_matmul.wmma.accumulator.sharedax2tileSpatial', 
 'batch_matmul.wmma.accumulatorb.ctileAll', 'batch_matmul.wmma.accumulatori.ctileAll', 'batch_matmul.wmma.accumulatorj.ctileAll', 
 'batch_matmul.wmma.accumulatorktileAll', 
 'batch_matmul.wmma.accumulatorb.c.innertileAll', 'batch_matmul.wmma.accumulatori.c.innertileAll', 
 'batch_matmul.wmma.accumulatorj.c.innertileAll', 
 'batch_matmul.wmma.accumulatork.innertileAll', 
 'batch_matmul.wmma.accumulatorb.c.inner.innertileAll', 'batch_matmul.wmma.accumulatori.c.inner.innertileAll', 'batch_matmul.wmma.accumulatorj.c.inner.innertileAll', 'batch_matmul.wmma.accumulatork.inner.innertileAll', 
 'B.shared_ax2', 'B.shared_offset', 'B.shared_vectorize', 
 'A.shared_ax2', 'A.shared_offset', 'A.shared_vectorize']







c1d             38             ['wmma_m', 'wmma_k', 'wmma_n', 'C.wmma.accumulator_shared_pos', 'C.wmma.accumulator_local_pos', 'C_shared_pos', 'C.wmma.accumulator.shared_local_pos', 'output_vectorize', 'threadIdx.x', 'threadIdx.y', 'C_unroll_pragma', 'Ci.innertileSpatial', 'Cj.innertileSpatial', 'Ci.inner.innertileSpatial', 'Cj.inner.innertileSpatial', 'Ci.inner.inner.innertileSpatial', 'Cj.inner.inner.innertileSpatial', 'Ci.inner.inner.inner.innertileSpatial', 'Cj.inner.inner.inner.innertileSpatial', 'C.wmma.accumulator.shared_ax1', 'C.wmma.accumulator.shared_offset', 'C.wmma.accumulator.sharedax0tileSpatial', 'C.wmma.accumulator.sharedax1tileSpatial', 'C.wmma.accumulatori.ctileAll', 'C.wmma.accumulatorj.ctileAll', 'C.wmma.accumulatorktileAll', 'C.wmma.accumulatori.c.innertileAll', 'C.wmma.accumulatorj.c.innertileAll', 'C.wmma.accumulatork.innertileAll', 'C.wmma.accumulatori.c.inner.innertileAll', 'C.wmma.accumulatorj.c.inner.innertileAll', 'C.wmma.accumulatork.inner.innertileAll', 'A.shared_ax1', 'A.shared_offset', 'A.shared_vectorize', 'B.shared_ax1', 'B.shared_offset', 'B.shared_vectorize']
c2d             38             ['wmma_m', 'wmma_k', 'wmma_n', 'C.wmma.accumulator_shared_pos', 'C.wmma.accumulator_local_pos', 'C_shared_pos', 'C.wmma.accumulator.shared_local_pos', 'output_vectorize', 'threadIdx.x', 'threadIdx.y', 'C_unroll_pragma', 'Ci.innertileSpatial', 'Cj.innertileSpatial', 'Ci.inner.innertileSpatial', 'Cj.inner.innertileSpatial', 'Ci.inner.inner.innertileSpatial', 'Cj.inner.inner.innertileSpatial', 'Ci.inner.inner.inner.innertileSpatial', 'Cj.inner.inner.inner.innertileSpatial', 'C.wmma.accumulator.shared_ax1', 'C.wmma.accumulator.shared_offset', 'C.wmma.accumulator.sharedax0tileSpatial', 'C.wmma.accumulator.sharedax1tileSpatial', 'C.wmma.accumulatori.ctileAll', 'C.wmma.accumulatorj.ctileAll', 'C.wmma.accumulatorktileAll', 'C.wmma.accumulatori.c.innertileAll', 'C.wmma.accumulatorj.c.innertileAll', 'C.wmma.accumulatork.innertileAll', 'C.wmma.accumulatori.c.inner.innertileAll', 'C.wmma.accumulatorj.c.inner.innertileAll', 'C.wmma.accumulatork.inner.innertileAll', 'A.shared_ax1', 'A.shared_offset', 'A.shared_vectorize', 'B.shared_ax1', 'B.shared_offset', 'B.shared_vectorize']
c3d             38             ['wmma_m', 'wmma_k', 'wmma_n', 'C.wmma.accumulator_shared_pos', 'C.wmma.accumulator_local_pos', 'C_shared_pos', 'C.wmma.accumulator.shared_local_pos', 'output_vectorize', 'threadIdx.x', 'threadIdx.y', 'C_unroll_pragma', 'Ci.innertileSpatial', 'Cj.innertileSpatial', 'Ci.inner.innertileSpatial', 'Cj.inner.innertileSpatial', 'Ci.inner.inner.innertileSpatial', 'Cj.inner.inner.innertileSpatial', 'Ci.inner.inner.inner.innertileSpatial', 'Cj.inner.inner.inner.innertileSpatial', 'C.wmma.accumulator.shared_ax1', 'C.wmma.accumulator.shared_offset', 'C.wmma.accumulator.sharedax0tileSpatial', 'C.wmma.accumulator.sharedax1tileSpatial', 'C.wmma.accumulatori.ctileAll', 'C.wmma.accumulatorj.ctileAll', 'C.wmma.accumulatorktileAll', 'C.wmma.accumulatori.c.innertileAll', 'C.wmma.accumulatorj.c.innertileAll', 'C.wmma.accumulatork.innertileAll', 'C.wmma.accumulatori.c.inner.innertileAll', 'C.wmma.accumulatorj.c.inner.innertileAll', 'C.wmma.accumulatork.inner.innertileAll', 'A.shared_ax1', 'A.shared_offset', 'A.shared_vectorize', 'B.shared_ax1', 'B.shared_offset', 'B.shared_vectorize']        
dil            59              ['wmma_m', 'wmma_k', 'wmma_n', 'out.wmma.accumulator_shared_pos', 'out.wmma.accumulator_local_pos', 'out_shared_pos', 'out.wmma.accumulator.shared_local_pos', 'out_unroll_pragma', 'outnn.innertileSpatial', 'outyy.innertileSpatial', 'outxx.innertileSpatial', 'outff.innertileSpatial', 'outnn.inner.innertileSpatial', 'outyy.inner.innertileSpatial', 'outxx.inner.innertileSpatial', 'outff.inner.innertileSpatial', 'threadIdx.y', 'outnn.inner.inner.innertileSpatial', 'outyy.inner.inner.innertileSpatial', 'outxx.inner.inner.innertileSpatial', 'outff.inner.inner.innertileSpatial', 'threadIdx.x', 'outnn.inner.inner.inner.innertileSpatial', 'outyy.inner.inner.inner.innertileSpatial', 'outxx.inner.inner.inner.innertileSpatial', 'outff.inner.inner.inner.innertileSpatial', 'out.wmma.accumulator.shared_ax3', 'out.wmma.accumulator.shared_offset', 'out.wmma.accumulator.sharedax0tileSpatial', 'out.wmma.accumulator.sharedax1tileSpatial', 'out.wmma.accumulator.sharedax2tileSpatial', 'out.wmma.accumulator.sharedax3tileSpatial', 'out.wmma.accumulatornn.ctileAll', 'out.wmma.accumulatoryy.ctileAll', 'out.wmma.accumulatorxx.ctileAll', 'out.wmma.accumulatorff.ctileAll', 'out.wmma.accumulatorrytileAll', 'out.wmma.accumulatorrxtileAll', 'out.wmma.accumulatorrctileAll', 'out.wmma.accumulatornn.c.innertileAll', 'out.wmma.accumulatoryy.c.innertileAll', 'out.wmma.accumulatorxx.c.innertileAll', 'out.wmma.accumulatorff.c.innertileAll', 'out.wmma.accumulatorry.innertileAll', 'out.wmma.accumulatorrx.innertileAll', 'out.wmma.accumulatorrc.innertileAll', 'out.wmma.accumulatornn.c.inner.innertileAll', 'out.wmma.accumulatoryy.c.inner.innertileAll', 'out.wmma.accumulatorxx.c.inner.innertileAll', 'out.wmma.accumulatorff.c.inner.innertileAll', 'out.wmma.accumulatorry.inner.innertileAll', 'out.wmma.accumulatorrx.inner.innertileAll', 'out.wmma.accumulatorrc.inner.innertileAll', 'filter.shared_ax3', 'filter.shared_offset', 'filter.shared_vectorize', 'PaddedInput.shared_ax3', 'PaddedInput.shared_offset', 'PaddedInput.shared_vectorize']
gemm           37               ['wmma_m', 'wmma_k', 'wmma_n', 'dense.wmma.accumulator_shared_pos', 'dense.wmma.accumulator_local_pos', 'dense_shared_pos', 'dense.wmma.accumulator.shared_local_pos', 'dense_unroll_pragma', 

'densei.innertileSpatial', 'densej.innertileSpatial', 'densei.inner.innertileSpatial', 'densej.inner.innertileSpatial', 'threadIdx.y', 'densei.inner.inner.innertileSpatial', 'densej.inner.inner.innertileSpatial', 'threadIdx.x', 'densei.inner.inner.inner.innertileSpatial', 'densej.inner.inner.inner.innertileSpatial', 

'dense.wmma.accumulator.shared_ax1', 'dense.wmma.accumulator.shared_offset', 'dense.wmma.accumulator.sharedax0tileSpatial', 'dense.wmma.accumulator.sharedax1tileSpatial', 

'dense.wmma.accumulatori.ctileAll', 'dense.wmma.accumulatorj.ctileAll', 'dense.wmma.accumulatorktileAll', 'dense.wmma.accumulatori.c.innertileAll', 'dense.wmma.accumulatorj.c.innertileAll', 'dense.wmma.accumulatork.innertileAll', 'dense.wmma.accumulatori.c.inner.innertileAll', 'dense.wmma.accumulatorj.c.inner.innertileAll', 'dense.wmma.accumulatork.inner.innertileAll',

'B.shared_ax1', 'B.shared_offset', 'B.shared_vectorize', 'A.shared_ax1', 'A.shared_offset', 'A.shared_vectorize']         
gemv           37               ['wmma_m', 'wmma_k', 'wmma_n', 'dense.wmma.accumulator_shared_pos', 'dense.wmma.accumulator_local_pos', 'dense_shared_pos', 'dense.wmma.accumulator.shared_local_pos', 'dense_unroll_pragma', 'densei.innertileSpatial', 'densej.innertileSpatial', 'densei.inner.innertileSpatial', 'densej.inner.innertileSpatial', 'threadIdx.y', 'densei.inner.inner.innertileSpatial', 'densej.inner.inner.innertileSpatial', 'threadIdx.x', 'densei.inner.inner.inner.innertileSpatial', 'densej.inner.inner.inner.innertileSpatial', 'dense.wmma.accumulator.shared_ax1', 'dense.wmma.accumulator.shared_offset', 'dense.wmma.accumulator.sharedax0tileSpatial', 'dense.wmma.accumulator.sharedax1tileSpatial', 'dense.wmma.accumulatori.ctileAll', 'dense.wmma.accumulatorj.ctileAll', 'dense.wmma.accumulatorktileAll', 'dense.wmma.accumulatori.c.innertileAll', 'dense.wmma.accumulatorj.c.innertileAll', 'dense.wmma.accumulatork.innertileAll', 'dense.wmma.accumulatori.c.inner.innertileAll', 'dense.wmma.accumulatorj.c.inner.innertileAll', 'dense.wmma.accumulatork.inner.innertileAll', 'BPad.shared_ax1', 'BPad.shared_offset', 'BPad.shared_vectorize', 'A.shared_ax1', 'A.shared_offset', 'A.shared_vectorize']
scan           37               ['wmma_m', 'wmma_k', 'wmma_n', 'dense.wmma.accumulator_shared_pos', 'dense.wmma.accumulator_local_pos', 'dense_shared_pos', 'dense.wmma.accumulator.shared_local_pos', 'dense_unroll_pragma', 'densei.innertileSpatial', 'densej.innertileSpatial', 'densei.inner.innertileSpatial', 'densej.inner.innertileSpatial', 'threadIdx.y', 'densei.inner.inner.innertileSpatial', 'densej.inner.inner.innertileSpatial', 'threadIdx.x', 'densei.inner.inner.inner.innertileSpatial', 'densej.inner.inner.inner.innertileSpatial', 'dense.wmma.accumulator.shared_ax1', 'dense.wmma.accumulator.shared_offset', 'dense.wmma.accumulator.sharedax0tileSpatial', 'dense.wmma.accumulator.sharedax1tileSpatial', 'dense.wmma.accumulatori.ctileAll', 'dense.wmma.accumulatorj.ctileAll', 'dense.wmma.accumulatorktileAll', 'dense.wmma.accumulatori.c.innertileAll', 'dense.wmma.accumulatorj.c.innertileAll', 'dense.wmma.accumulatork.innertileAll', 'dense.wmma.accumulatori.c.inner.innertileAll', 'dense.wmma.accumulatorj.c.inner.innertileAll', 'dense.wmma.accumulatork.inner.innertileAll', 'B.shared_ax1', 'B.shared_offset', 'B.shared_vectorize', 'A.shared_ax1', 'A.shared_offset', 'A.shared_vectorize']
t2d            38               ['wmma_m', 'wmma_k', 'wmma_n', 'C.wmma.accumulator_shared_pos', 'C.wmma.accumulator_local_pos', 'C_shared_pos', 'C.wmma.accumulator.shared_local_pos', 'output_vectorize', 'threadIdx.x', 'threadIdx.y', 'C_unroll_pragma', 'Ci.innertileSpatial', 'Cj.innertileSpatial', 'Ci.inner.innertileSpatial', 'Cj.inner.innertileSpatial', 'Ci.inner.inner.innertileSpatial', 'Cj.inner.inner.innertileSpatial', 'Ci.inner.inner.inner.innertileSpatial', 'Cj.inner.inner.inner.innertileSpatial', 'C.wmma.accumulator.shared_ax1', 'C.wmma.accumulator.shared_offset', 'C.wmma.accumulator.sharedax0tileSpatial', 'C.wmma.accumulator.sharedax1tileSpatial', 'C.wmma.accumulatori.ctileAll', 'C.wmma.accumulatorj.ctileAll', 'C.wmma.accumulatorktileAll', 'C.wmma.accumulatori.c.innertileAll', 'C.wmma.accumulatorj.c.innertileAll', 'C.wmma.accumulatork.innertileAll', 'C.wmma.accumulatori.c.inner.innertileAll', 'C.wmma.accumulatorj.c.inner.innertileAll', 'C.wmma.accumulatork.inner.innertileAll', 'A.shared_ax1', 'A.shared_offset', 'A.shared_vectorize', 'B.shared_ax1', 'B.shared_offset', 'B.shared_vectorize']   


'wmma_m', 
dense_wmma_accumulator_shared_ax0_inner_inner
dense_wmma_accumulator_i_c_inner_inner_inner_inner
A_shared_wmma_matrix_a_ax0_inner


'wmma_k', 
dense_wmma_accumulator_k_inner_inner_inner_inner
B_shared_wmma_matrix_b_ax1_inner

'wmma_n', 
dense_wmma_accumulator_shared_ax1_inner_inner
dense_wmma_accumulator_j_c_inner_inner_inner_inner
B_shared_wmma_matrix_b_ax0_inner


'dense_wmma_accumulator_shared_pos', 
'dense_wmma_accumulator_local_pos',
'dense_shared_pos', 
'dense_wmma_accumulator_shared_local_pos', 
'dense_unroll_pragma', 


 
'densei_innertileSpatial',
dense_i_inner_outer    blockIdx_x
 
 
'densej_innertileSpatial', 
dense_j_inner_outer    blockIdx_x
  
  
'densei_inner_innertileSpatial',
dense_i_inner_inner_outer  threadIdx_y
  
'densej_inner_innertileSpatial',
dense_j_inner_inner_outer
   
   
'threadIdx_y',      dense_wmma_accumulator_sharedax0tileSpatial, dense_wmma_accumulator_sharedax1tileSpatial
threads
    
'densei_inner_inner_innertileSpatial',
dense_i_inner_inner_inner_outer         threadIdx_x
    
'densej_inner_inner_innertileSpatial',
dense_j_inner_inner_inner_outer
     
'threadIdx_x', 
threads


'densei_inner_inner_inner_innertileSpatial',
dense_i_inner_inner_inner_inner_outer

'densej_inner_inner_inner_innertileSpatial',
dense_j_inner_inner_inner_inner_outer




'dense_wmma_accumulator_shared_ax1',        dense_wmma_accumulator_shared_align_size  dense_wmma_accumulator_shared_ax1_outer, dense_wmma_accumulator_shared_ax1_inner
012345

'dense_wmma_accumulator_shared_offset',     dense_wmma_accumulator_shared_align_size
012345

'dense_wmma_accumulator_sharedax0tileSpatial', 
dense_wmma_accumulator_shared_ax0_outer     A_shared_ax0  threadIdx.y

'dense_wmma_accumulator_sharedax1tileSpatial', 
dense_wmma_accumulator_shared_ax1_outer     B_shared_ax0  threadIdx.y

'dense_wmma_accumulatori_ctileAll', 
dense_wmma_accumulator_i_c_outer


'dense_wmma_accumulatorj_ctileAll', 
dense_wmma_accumulator_j_c_outer


'dense_wmma_accumulatorktileAll', 
dense_wmma_accumulator_k_outer


'dense_wmma_accumulatori_c_innertileAll', 
dense_wmma_accumulator_i_c_inner_outer


'dense_wmma_accumulatorj_c_innertileAll', 
dense_wmma_accumulator_j_c_inner_outer


'dense_wmma_accumulatork_innertileAll', 
dense_wmma_accumulator_k_inner_outer


'dense_wmma_accumulatori_c_inner_innertileAll', 
dense_wmma_accumulator_i_c_inner_inner_outer

'dense_wmma_accumulatorj_c_inner_innertileAll', 
dense_wmma_accumulator_j_c_inner_inner_outer

'dense_wmma_accumulatork_inner_innertileAll', 
dense_wmma_accumulator_k_inner_inner_outer


'B_shared_ax1',         dense_wmma_accumulator_k_inner      dense_wmma_accumulator_k_inner_inner        dense_wmma_accumulator_k_inner_inner_inner      dense_wmma_accumulator_k_inner_inner_inner_inner  B_shared_align_size
0123
B_shared_ax0_ax1_fused


'B_shared_offset',  B_shared_align_size
012345


'B_shared_vectorize', 
0123

'A_shared_ax1',     dense_wmma_accumulator_k_inner      dense_wmma_accumulator_k_inner_inner    dense_wmma_accumulator_k_inner_inner_inner      dense_wmma_accumulator_k_inner_inner_inner_inner        A_shared_align_size
01234
A_shared_ax0_ax1_fused


'A_shared_offset', A_shared_align_size
012345


'A_shared_vectorize'
01234

dense
start             blockIdx.x = densei_innertileSpatial * densej_innertileSpatial
unrollPragma    得出一个规则
densei_innertileSpatial > densei_inner_innertileSpatial > densei_inner_inner_innertileSpatial > densei_inner_inner_inner_innertileSpatial
tileBlock       dense_i_outer=1  dense_i_inner=1024    dense_j_outer=1  dense_j_inner=1024
tileThread      (densei_innertileSpatial) dense_i_inner_outer       (densej_innertileSpatial) dense_j_inner_outer
(dense_i_inner_outer -> dense_i_inner_inner) -> dense_i_inner_outer_j_inner_outer_fused -> (dense_i_inner_inner_outer ->dense_i_inner_inner_inner) -> dense_i_inner_inner_outer_j_inner_inner_outer_fused -> (dense_i_inner_inner_inner_outer -> dense_i_inner_inner_inner_inner) -> dense_i_inner_inner_inner_outer_j_inner_inner_inner_outer_fused -> (dense_i_inner_inner_inner_inner_outer -> dense_i_inner_inner_inner_inner_inner) -> dense_i_inner_inner_inner_inner_inner_j_inner_inner_inner_inner_inner_fused -> dense_vectorize
tileWarp
vectorize
finish

dense_mem_accumulator_shared_mem_size=dense_wmma_accumulator_shared_ax0+dense_wmma_accumulator_shared_align_size
dense_wmma.accumulator.shared        
start            dense_wmma_accumulator_shared_ax0 (dense_i_inner, dense_i_inner_inner, dense_i_inner_inner_inner, dense_i_inner_inner_inner_inner_inner)
storageAlign     
tileThread       dense_wmma_accumulator_shared_ax0 (dense_wmma_accumulator_sharedax0tileSpatial) -> (dense_wmma_accumulator_shared_ax0_outer, dense_wmma_accumulator_shared_ax0_inner) -> dense_wmma_accumulator_shared_ax0_outer_ax1_outer_fused
tensorcoreStore dense_wmma_accumulator_shared_ax0_inner -> (dense_wmma_accumulator_shared_ax0_inner_outer, dense_wmma_accumulator_shared_ax_inner_inner)
finish


dense_wmma.accumulator
start           dense_wmma_accumulator_i_c (dense_wmma_accumulator_shared_ax0_inner)  dense_wmma_accumulator_j_c (dense_wmma_accumulator_shared_ax1_inner)
generalTile     dense_wmma_accumulator_i_c -> (dense_wmma_accumulator_i_c_outer, dense_wmma_accumulator_i_c_inner) -> (dense_wmma_accumulator_i_c_inner_outer, dense_wmma_j_c_inner_inner) -> (dense_wmma_accumulator_i_c_inner_inner_outer, dense_wmma_accumulator_i_c_inner_inner_inner)
tensorcore     dense_wmma_accumulator_i_c_inner_inner_inner -> (dense_wmma_accumulator_i_c_inner_inner_inner_outer, dense_wmma_accumulator_i_c_inner_inner_inner_inner)  dense_wmma_accumulator_i_c_inner_inner_inner_inner=wmma_m
finish

B_shared_wmma_matrix_b_ax0、B_shared_wmma_matrix_b_ax1
B.shared.wmma.matrix_b
start               B_shared_tmp_ax0 (dense_wmma_accumulator_j_c_inner, dense_wmma_accumulator_j_c_inner_inner, dense_wmma_accumualtor_j_c_inner_inner_inner, dense_wmma_accumulator_j_c_inner_inner_inner_inner)
tensorcoreLoadB    B_shared_ax0 -> (B_shared_tmp_ax0, dense_wmma_accumulator_sharedax1tileSpatial)
finish

B.shared
start           B_shared_ax1 (dense_wmma_accumulator_k_inner, dense_wmma_accumulator_k_inner_inner, dense_wmma_accumulator_k_inner_inner_inner, dense_wmma_accumulator_k_inner_inner_inner_inner)
defaultsharedLoadSchedOP B_shared_offset -> 【B.shared_ax1, B_shared_offset】-> B_shared_align_size B_shared_vectorize  B.shared_ax0_ax1_fused -> (B.shared_ax0, B.shared_ax1)

A.shared.wmma.matrix_a
start           A_shared_wmma_matrix_a_ax0 (dense_wmma_matrix_i_c_inner, dense_wmma_matrix_i_c_inner_inner, dense_wmma_matrix_i_c_inner_inner_inner, dense_wmma_matrix_i_c_inner_inner_inner_inner)、A_shared_wmma_matrix_a_ax1 (dense_wmma_accumulator_k_inner, dense_wmma_accumulator_k_inner_inner, dense_wmma_accumulator_k_inner_inner_inner, dense_wmma_accumulator_k_inner_inner_inner_inner)    
tensorcoreLoadA  A_shared_wmma_matrix_a_ax0 -> (A_shared_wmma_matrix_a_ax0_outer, A_shared_wmma_matrix_a_ax0_inner[wmma_m]) A_shared_wmma_matrix_a_ax1 -> (A_shared_wmma_matrix_a_ax1_outer, A_shared_wmma_matrix_a_ax1_inner[wmma_k])
finish

A.shared
start           A_shared_tmp_ax0(dense_wmma_accumulator_i_c_inner, dense_wmma_matrix_i_c_inner_inner, dense_wmma_matrix_i_c_inner_inner_inner, dense_wmma_matrix_i_c_inner_inner_inner_inner)
                A_shared_ax1(dense_wmma_accumulator_k_inner, dense_wmma_accumulator_k_inner_inner, dense_wmma_accumulator_k_inner_inner_inner, dense_wmma_accumulator_k_inner_inner_inner_inner)
defaultsharedLoadSchedOP A_shared_offset -> 【A.shared_ax1, B_shared_offset】 -> A_shared_align_size A_shared_ax0_ax1_fused -> (A.shared_ax0, A_shared_ax1) A_shared_vectorize

A 
start   
finish  

threads -> (threadIdx_x, threadIdx_y)
A_shared_shared_mem_size -> (A_shared_ax0, A_shared_align_size)
B_shared_shared_mem_size -> (B_shared_ax0, B_shared_align_size)
dense_wmma_accumulator_shared_shared_mem_size -> (dense_wmma_accumulator_shared_ax0, dense_wmma_accumulator_shared_align_size)


分析失败的原因
0.0,16,16,16,1,2,1,0,5,4,16,16,1,16,1,32,32,16,1,64,24,16,1,1,1,32,1,1,1,1,1,1,32,24,2,32,24,2


'wmma_m', 'wmma_k', 'wmma_n', 
'dense.wmma.accumulator_shared_pos', 'dense.wmma.accumulator_local_pos', 
'dense_shared_pos', 'dense.wmma.accumulator.shared_local_pos', 'dense_unroll_pragma', 

'densei.innertileSpatial', 'densej.innertileSpatial', 'densei.inner.innertileSpatial', 'densej.inner.innertileSpatial', 'threadIdx.y', 
'densei.inner.inner.innertileSpatial', 'densej.inner.inner.innertileSpatial', 'threadIdx.x', 'densei.inner.inner.inner.innertileSpatial', 'densej.inner.inner.inner.innertileSpatial', 

'dense.wmma.accumulator.shared_ax1', 'dense.wmma.accumulator.shared_offset', 
'dense.wmma.accumulator.sharedax0tileSpatial', 'dense.wmma.accumulator.sharedax1tileSpatial', 

'dense.wmma.accumulatori.ctileAll', 'dense.wmma.accumulatorj.ctileAll', 'dense.wmma.accumulatorktileAll', 
'dense.wmma.accumulatori.c.innertileAll', 'dense.wmma.accumulatorj.c.innertileAll', 'dense.wmma.accumulatork.innertileAll', 
'dense.wmma.accumulatori.c.inner.innertileAll', 'dense.wmma.accumulatorj.c.inner.innertileAll', 'dense.wmma.accumulatork.inner.innertileAll',

'B.shared_ax1', 'B.shared_offset', 'B.shared_vectorize', 'A.shared_ax1', 'A.shared_offset', 'A.shared_vectorize'

77,8.906950110460233,   16,16,16, 3,3, 1,0,5, 4,16,16,1,16, 1,32,32,16,1, 64,8, 16,1, 1,4,32, 1,1,1, 1,1,1, 16,0,1, 16,24,2
1190,10.284423285696588,16,16,16, 0,1, 1,0,5, 4,16,16,1,16, 1,32,32,16,1, 64,0, 16,1, 1,1,32, 1,1,1, 1,4,2, 32,24,8, 32,8,8