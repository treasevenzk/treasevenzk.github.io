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


