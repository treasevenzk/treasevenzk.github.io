---
layout:     post
title:      code reproduction
subtitle:   Ansor-AF-DS
date:       2025-06-04
author:     Treaseven
header-img: img/bg2.jpg
catalog: true
tags:
    - debug
---

include
auto_scheduler: cost_model.h、feature.h、measure.h、measure_record.h
tir: analysis.h

src
auto_scheduler: cost_model.cc、feature.cc、measure.cc、measure_record.cc
auto_scheduler/search_policy: sketch_policy.cc、sketch_policy_rules.cc
search_policy: sketch_analysis.cc、sketch_analysis.h、sketch_policy.cc
tir: verify_gpu_code.cc


python
auto_scheduler: __init__.py、dispatcher.py、feature.py、task_scheduler.py
auto_scheduler/cost_model: __init__.py、cost_model.py


python3 test/batch_conv2d_cuda.py yolo 128 48 0 5 64 0.6


splitMeta类
step_id、adjust_stage_id、problem_size、origin_itr、tile_sizes、parallel

TDx_access_info结构体
buffer_name、buffer_touch_size、local_threadx_val_map


GetPerStoreOurFeature函数
总线程块数量: num_TB = grid_dimx*grid_dimy*grid_dimz
每个线程块的warp数量: num_warp = block_dimx*block_dimy*block_dimz / 32
缓冲区访问的数据大小: touch_size
每个缓冲区访问的warp级事务数: warp_trans = (xacc_fea.touch_size/block_dimx)*num_warp
输入数据共享内存事务: shared_trans_input
卷积核共享内存事务: shared_trans_kernel
占用率: occupancy = 实际warp数量 / 最大warp数
feature: 1、global_trans、shared_trans、occupancy、num_TB、shared_trans_input、shared_trans_kernel


MathOpCounter、BufferAccessExtractor、CoefficientExtractor、BCExtractor、PerStoreFeatureExtractor、BufferExtractor、ReuseExtractor、IndexExtractor


GetPerStoreOurFeature_bc函数
tdx_acc_info: 进行bank conflict分析所需的关键信息
tdx_acc_info.buffer_name: 不同类型的共享内存缓冲区
tdx_acc_info.buffer_touch_size: 内存事务的数量
tdx_acc_info.local_threads_val_map: 线程映射模式，决定bank conflict发生模式


generateSplitMeta函数: 从TVM的调度状态中提取所有分割操作的元数据，用于后续的特征分析和性能建模
共享内存阶段的split通常不影响主要的性能特征


features_extracted: global_trans、est_occupancy、shared_trans、num_TB


extract_features函数
parallel_data_map: 并行维度
true_reduction_data_map: 真正的规约维度
stencil_reduction_data_map: 模板规约维度
ILP: 指令级并行度
WLP: warp级并行度
WLP_REG/WLP_SM: 寄存器/共享内存限制的WLP
Concurrent_estimate: 并发估计
OI_Global/OI_Shared: 全局/共享内存的操作强度


hardware_params = auto_scheduler.HardwareParams(target, num_cores, max_shared_memory_per_block, max_threads_per_block, max_vthread_extent, vector_unit_bytes, cache_line_bytes)
task = auto_scheduler.SearchTask(func, args, target, hardware_params)
tune_options = auto_scheduler.TuningOptions(num_measure_trials, measure_callbacks, verbose)
cost_model = auto_scheduler.XGBModel()
search_policy = auto_scheduler.SketchPolicy(task, program_cost_model, params)
task.tune(tune_options, search_policy)


Ansor_DS
SearchONeRoundDGD
GenerateSketches() → GenerateSplitMeta → GetFactorInfo → computeSMTileSize → SampleCUDAInitPopulation → DGD_Search


GenerateSplitMeta函数: 负责从调度状态中提取分割操作的元数据信息
cache read: 在目标stage之后插入
cache write: 在目标stage之前插入


GetFactorInfo函数: 负责计算每个循环维度所有可能因子的核心组件
CUDA多级Tiling层次结构
并行维度: [Grid, ThreadBlock, Thread, Vectorize, Remainder]
串行维度: [Outer, Inner, Remainder]
SplitFactorizationMemo: 创建银子分解备忘录对象
ComputeSMTileSize: 根据寄存器级别的tile因子计算共享内存层次的tile大小

GPU内存层次结构
全局内存(Global Memory) ← 最大容量，最慢访问
    ↑
共享内存(Shared Memory) ← compouteSMTileSize优化的层级
    ↑
寄存器(Register)        ← reg_tile_factors对应的层级

gen_neigbour_list
GetStateFactor → GenerateSplitMeta → ConfigKey2string → cur_config_len → getExcludedDims → getRemainingDims → generateMasks → MaskUpDownMutate → SampleUniquePopulation


DGD_Move: base_state(基础状态)、neighbour_scores(邻居状态的预测分数)、indices(邻居状态按分数排序的索引)、loal_path_neighbors(本地路径邻居状态数组)、visited(已访问的集合)、max_idx(最优邻居索引)、next_state(下一状态数组)、index(当前状态索引)、found_better(是否找到更好的状态)、measure(程序测量器)、window_size(窗口大小)、tolerant_score(容忍分数阈值)、global_best_gflops(全局最佳性能)、model_age(模型年龄)、total_inputs、total_results


state_to_string: 将一个state转换为字符串表示，基于tiling size和split meta信息，用于生成state的唯一标识符，便于去重和缓存



bash run_tests_times_conv.sh conv2d       cuda       3          128         48                  resnet      5       64              0.6
                             test_type   platform   run_time   sm_num      max_shared_memory    network   ntrials   init_states     threshold


python batch_conv2d_cuda.py resnet 68 48 -1 5 64 0.6


全局内存的向量化优势
CPU/GPU Core → L1 Cache → L2 Cache → DRAM
全局内存事务大小：典型事务(128字节，32个float值)，向量化可以充分利用事务带宽

共享内存访问机制
//共享内存的bank结构
Shared Memory Banks(典型配置: 32个bank)
Bank 0: [0, 32, 64, 96, ...]
Bank 1: [1, 33, 65, 97, ...]
Bank 2: [2, 34, 66, 98, ...]
...
Bank 31: [31, 63, 95, 127, ...]

共享内存不使用向量化
1.共享内存访问已经很快，向量化带来的收益微乎其微
2.向量化可能引入bank conflict问题
3.编译器对共享内存的向量化优化不如全局内存激进
4.实际的内存事务数更接近原始计算值


splitMeta {
public:
    int step_id;
    int adjust_stage_id;
    int problem_size;
    Iterator origin_itr;
    std::vector<int> tile_sizes;
    bool parallel;
}


sm_name_val_map         # 共享内存中各维度的大小
reg_novth_name_val_map  # 寄存器中各维度大小
reg_name_val_map        # 寄存器中各维度的总大小
reg_split_node          # 并行维度的分割节点
sm_split_node           # 归约维度的分割节点
grid_size               # 网格总大小
thread_block_size       # 线程块大小
registerUsed_per_thread # 每线程寄存器使用量
num_thread_per_block    # 每线程块线程数
output_reg              # 输出寄存器数量
sm_prod_reduction       # 共享内存归约维度乘积



调度变换步骤序列
transform_steps = [
  Step 0: 初始状态
  Step 1: split i: [1024] -> [32, 32]     # stage_id=0, iter_id=0
  Step 2: split j: [1024] -> [16, 64]     # stage_id=0, iter_id=1  
  Step 3: split k: [512] -> [8, 64]       # stage_id=0, iter_id=2
  Step 4: cache_read A -> A.shared        # 创建新stage，stage_id=1
  Step 5: compute_at A.shared -> C[k]     # 将A.shared绑定到C的k循环
  Step 6: cache_read B -> B.shared        # 创建新stage，stage_id=2  
  Step 7: compute_at B.shared -> C[k]     # 将B.shared绑定到C的k循环
  Step 8: pragma "auto_unroll_max_step$16" # 设置展开因子
  Step 9: cache_write C -> C.local        # 创建新stage，stage_id=3
  Step 10: compute_at C.local -> C[j]     # 将C.local绑定到C的j循环
]
当前stage结构
stages = [
  stage[0]: C (原始计算stage)
  stage[1]: A.shared (cache read stage)  
  stage[2]: B.shared (cache read stage)
  stage[3]: C.local (cache write stage)
]

C[i, j] = A[i, k] * B[k, j]
A[1024, 512] B[512, 1024] C[1024, 1024]


v_splitMeta_info = [
  splitMeta_i: {
    origin_itr: "i", parallel: true,
    problem_size: 1024,
    tile_sizes: [4, 4, 16, 2, 2]  # [grid, coarse, block, vthread, thread]
  },
  splitMeta_j: {
    origin_itr: "j", parallel: true, 
    problem_size: 1024,
    tile_sizes: [2, 8, 32, 2, 1]  # [grid, coarse, block, vthread, thread]
  },
  splitMeta_k: {
    origin_itr: "k", parallel: false,
    problem_size: 512, 
    tile_sizes: [8, 32, 2]         # [outer_SM, inner_thread]
  }
]

*features = [
    12800.0, # global_trans
    0.75,    # est_occupancy
    8192.0   # shared_trans
    8        # num_TB
]


output_FVI = "j"                // 最快变化索引
par_order_array = ["j", "i"]    // 轴顺序

index_extract = {
    parallel_index: ["i", "j"], // 并行维度
    reduction_index: ["k"],     // 归约维度
    stencil_index: []           // 模板维度
}

true_reduction_index = ["k"]

ret_state = {
    stages: [
        stage[0]: C(主计算)
        stage[1]: A.shared (缓存读取)
        stage[2]: B.shared (缓存读取)
        stage[3]: C.local (缓存写入)
    ]
}

并行维度数据
parallel_data_map = {
    "i": {reg: 16, pz: 1024, tb: 16}, 寄存器、问题大小、线程块信息
    "j": {reg: 8, pz:1024, tb: 32},
}

共享内存映射
sm_name_val_map = {
    "i": 256,
    "j": 512,
    "k": 64
}

寄存器映射
reg_name_val_map = {
    "i": 16,
    "j": 16,
    "k": 1
}

reg_novth_name_val_map = {
    "i": 4,
    "j": 2,
}

归约维度数据
true_reduction_data_map = {
    "k": {sm: 64, pz: 512}  共享内存、问题大小信息
}


grid tile = 4: 分配给4个SM
线程粗粒度 = 2: 每个线程处理2批数据
线程块 = 2： 每个SM启动2个线程
寄存器tile1 = 2: 寄存器分2组
寄存器tile2 = 2： 每组处理两个元素


Grid Tile: 充分利用多个SM，避免某些SM空闲
线程块: 在SM内创建足够的并行度，隐藏内存延迟
线程粗粒度: 增加每个线程的工作量，提高指令级并行度
寄存器Tile: 在最快的存储层次进行密集计算，支持向量化

                             test_type platform run_time sm_num max_shared_memory network ntrials init_state specify_pz
bash run_tests_times_conv.sh conv2d cuda 3 128 48 resnet 1000 64 0.6



并行循环: 循环的不同迭代之间没有数据依赖关系，可以同时执行
非并行循环: 循环的迭代之间有数据依赖关系，必须按顺序执行

```
for (int i = 0; i < 1024; i++) {
    for (int j = 0; j < 1024; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}

__global__ void matrix_add(float* A, float* B, float* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 1024 && j < 1024) {
        C[i*1024 + j] = A[i*1024 + j] + B[i*1024 + j];
    }
}


float sum = 0.0f;
for (int i = 0; i < 1024; i++) {
    sum += array[i];
}

float sum = 0.0f;
for(int i_outer = 0; i_outer < 64; i_outer++) {     // 外层分块
    for (int i_inner = 0; i_inner < 16; i_inner++) {    // 内层向量化
        int i = i_outer * 16 + i_inner++;
        sum += array[i];
    }
}

```

|特性|并行循环|非并行循环|
|:---:|:---:|:---:|
|数据依赖|无依赖|有依赖|
|执行方式|可同时执行|必须顺序执行|
|优化目标|最大化并行度|优化缓存和向量化|
|目标硬件|GPU，多核CPU|单核CPU、缓存优化|
|TVM配置|5个瓦片->4个参数|3个瓦片->2个参数|
|典型应用|矩阵运算的空间循环|归约操作、累积运算|
