---
layout:     post
title:      PrunerCode
subtitle:   Code Reproduction
date:       2025-03-26
author:     Treaseven
header-img: img/bg20.jpg
catalog: true
tags:
    - debug
---
on-line cost model
Pruner: 1.69ms      Used time 5563s     Estimated total latency: 1.476ms
MoA-Pruner: 1.70ms  Used time 4978s     Estimated total latency: 1.457ms
Ansor: 3.96ms       Used time 6691s     Estimated total latency: 1.592ms


off-line cost model
Pruner-offline: 1.71ms Used time 4212s  Estimated total latency: 1.469ms
TensetMLP:  1.79ms     Used time 5469s  Estimated total latency: 1.621ms



```
源码修改
inlcude
tvm/auto_scheduler
feature_pam.h、feature_psa.h

python
tvm/auto_scheduler
cost_model/pam_model.py、psa_model.py
修改 dataset.py、feature.py、search_policy.py、task_scheduler.py


src
auto_scheduler
feature_pam.cc、feature_psa.cc
修改 search_policy sketch_policy.cc、sketch_policy.cc
```

测试脚本
search with Pruner
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model pam --target "cuda --model=a100" --psa a100_40

search with the MoA-Pruner
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model pam-siamese-update --load-model pam_k80_1500.pkl --target "cuda --model=a100" --psa a100_40

search with Ansor
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model mlp --target "cuda --model=a100"

auto_scheduler源码分析---ansor
Extract search tasks
auto_scheduler.extract_tasks

get_tuning_option
auto_scheduler.LocalRPCMeasureContext
auto_scheduler.TuningOptions
auto_scheduler.RecordToFile

Run search
auto_scheduler.TaskScheduler

evaluate results
local_search、default_search
from tvm.auto_scheduler.measure_record import load_records, save_records
from tvm.auto_scheduler.utils import decode_workload_key
from tvm.auto_scheduler.measure import MeasureInput
auto_scheduler.ApplyHistoryBest
tvm.transform.PassContext
relay.build
tvm.context
runtime.GraphModule


注册的全局函数
***feature_pam.cc***: GetPerStoreFeaturesFromStatePAM、GetPerStoreFeaturesFromMeasurePairsPAM
***feature_psa.cc***: GetPerStoreFeaturesFromStatePSA、GetPerStoreFeautresFromMeasurePairsPSA


***measure.cc***: EmptyBuilder、EmptyRunner

auto_scheduler
88+1521+174+1698+576+483+477+176+1880=7073
1764+975=2739

search_policy
126+124+1242+790+503=2785



从relay_integration.py文件中
grc.codegen -> GraphRuntimeCodegen.Codegen -> heads_ = VisitExpr(func->body) -> std::vector<GraphNodeRef> VisitExpr_(const CallNode* call_node) -> GraphAddCallNode -> CachedFunc lowered_func = (*pf1)(compile_engine_, key)


在compile_engine.cc文件中
relay.backend._CompileEngineLower -> self.Lower -> LowerInternal -> CreateSchedule -> ScheduleGetter.Create -> 调用python端的auto_scheduler.relay_integration.auto_scheduler_topi_compute

一个问题就是
self._mod = _build_module._GraphRuntimeCodegen()
self._init = self._mod["init"]
上面这两行代码里面会涉及到去调用C++运行文件graph_runtime_codegen.cc文件类GraphRuntimeCodegenModule中的GetFunction方法从而保证上面两行代码的正确性
这是因为在TVM的python里面tvm/_ffi/_ctypes/moudle.py文件中 Python Module类中def __getitem__(self, name): return self.get_function(name)



self.callbacks = [PrintTableInfo(), LogEstimatedLatency("total_latency.tsv")]

python中namedtuple和OrderedDict
namedtuple: 用于创建一个带有命名字段的元组，普通的元组只能通过索引访问其元素，元组是不可变的，创建后不能修改、添加或删除元素
OrderedDict: 特殊的字典类型，记住了键值对添加的顺序，普通的字典不保证元素的顺序


PAMModel
predict函数
get_per_store_features_from_states_pam -> _ffi_api.GetPerStoreFeaturesFromStatesPAM -> unpack_feature_apm -> PAMDataset.create_one_task -> load_task_data -> self.model.predict -> self._predict_a_dataset -> self._predict_a_task -> PAMDataLoader

update函数
PAMDataset.update_from_measure_pairs -> get_per_store_features_from_measure_pairs_pam -> _ffi_api.GetPerStoreFeaturesFromMeasurePairsPAM -> unpack_feature_pam -> self.model.fit_base -> self._fit_a_model -> self.register_new_task -> PAMDataLoader -> self.make_net -> PAMModule


auto_scheduler.GetPerStoreFeaturesFromStatesPAM
GetPerStoreFeatureFromStatesPAM -> ExtractAndPaddingFeaturePAM -> GetPerStoreFeaturesWorkerFuncPAM -> GetPerStoreFeaturePAM -> PerStoreFatureExtractorPAM -> KMP -> FeatureSetPAM -> AnnotationPosTypePAM -> BufferAccessTypePAM -> ReuseTypePAM -> BufferFeatureSetPAM -> BufferAccessFeaturePAM -> LocationTypePAM -> 

GetPerStoreFeatureworkerFuncPAM函数的作用
用于在TVM中为自动调度提取每个存储操作特征

GetPerStoreFeaturePAM函数的作用
从TVM语句中提取性能相关特征，这些特征会被用于性能模型，以预测TVM调度的执行时间

SerializeFeaturesPAM函数
将提取的特征数据序列化为字节数组，以便在C++和Python之间传递数据



GetPerStoreFeaturePAM
PerStoreFeatureExtractorPAM(类)、KMP、FeatureSetPAM(结构体)、AnnotationPosTypePAM(类)、BufferAccessTypePAM(类)、ReuseTypePAM(类)、BufferFeatureSetPAM(结构体)、BufferAccessFeaturePAM(结构体)、LocationTypePAM(类)、


ComputeRegionPAM、ComputeReusePAM、ComputeStridePAM、CoefficientExtractorPAM、BufferAccessExtractorPAM、MathOpCounterPAM(类)、GetLoopExtentPAM(函数)、GetAnnotationPosEncodingPAM、VarInExprPAM、BufferAccessPAM


feature_pam.cc文件
PerStoreFeatureExtractorPAM:
VisitStmt_(BufferRealizeNode):
StorageScope -> runtime::DefaultStorageRank/StorageScope::Create -> StmtExprVisitor::VisitStmt_(node) -> ExtractAllocationFeature

VisitStmt_(BufferStoreNode):
MathOpCounterPAM -> ExtractComputationFeature -> ExtractBufferAccessFeature -> ExtractArithmeticIntensityFeature -> ExtractOuterScopeFeature

VisitStmt_(ForNode):
GetLoopExtentPAM -> StmtExprVisitor::VisitStmt_

VisitStmt_(AttrStmtNode):
StmtExprVisitor::VisitStmt_

Pruner的网络模型架构
平常特征
self.segment_encoder

矩阵乘特征
self.gemm_encoder
self.attention

output = torch.cat([segment_sum, gemm_mha_output], dim=1)
self.fuse

decoder
self.norm
self.l0
self.l1
self.decoder


sketch_rules:
rule_add_cache_read_stage、rule_special_compute_location_gpu、rule_always_inline、rule_simplify_compute_with_const_tensor、rule_cross_thread_reduction、rule_add_cache_write_stage、rule_multi_level_tiling_with_fusion、rule_multi_level_tiling、rule_skip_stage

gdb --args python tune_network.py --network resnet_50 --n-trials 200 --cost-model pam --target "cuda --model=a100" --psa a100_40

arg.target_host: None      arg.result_file: results.tsv     arg.transfer_tune: None     args.search_type: default

network_args = {
    "network": resnet_50,
    "batch_size": 1
}

tuning_args = {
    "eval_only": false,
    "continue_tuning": false,
    "n_trials": 200
    "num_measures_per_round": 10
    "log_file": resnet_50-B1-cuda-a100.json
    "run_timeout": 25,
    "cost_model": pam,
    "load_model": None,
    "n_lines": None,
    "psa_model_type": a100_40
}


TaskScheduler
tasks: tasks        objective_func = lambda costs: sum(c * w for c, w in zip(costs, task_weights))      strategy: gradient      load_log_file: None     load_model_file: None          alpha: 0.2       beta: 2         gamma: 0.5      backward_window_size: 3
callbacks: [PrintTableInfo(), LogEstimateLatency("total_latency.tsv")]
task_cts = [0 for _ in range(len(self.tasks))]  记录任务i被调优多少次
task_best_cts = [0 for _ in range(len(self.tasks))] 记录任务i当前最佳延迟
task_costs_history = [[] for _ in range(len(self.tasks))] 记录任务i历史延迟记录
best_costs = 1e10 * np.ones(len(self.tasks)) 记录任务i最佳延迟
cur_score = self._compute_score(self.best_costs)
tune_option、measurer、search_policies、ct、best_ct、best_score、tic、num_measures_per_round: None
dead_tasks = set()

task_tags、tag_to_groud_id、group_task_ids、flop_cts

search_policy = 'sketch.pam'      psa_model_type = a100_40      search_policy_params = None     num_measure_per_round = 10      tune_option.verbose = 1     load_model_file = None      load_log_file = None        adaptive_training = false
disable_cost_model_update = false

cost_model = PAMModel(disable_update, few_shot_learning)       cost_model_psa = PSAModel(peak_performance=19490, glbmem_bandwidth=1555, vec_len=11, activate_blocks_per_sm=1, sm_nums=108, arm_sm_partition=4, arch_warp_size=32)
init_search_callbacks = [PreloadMeasuredStates(load_log_file)]
search_policies = [SketchPolicy(task, cost_model, cost_model_psa, params, verbose, init_search_callbacks) for task in tasks]

PAMModel
BufferNameMap、BufferCache
PerStoreFeatureExtractorPAM: VisitStmt_(AttrStmtNode)、VisitStmt_(BufferRealizeNode)



========== Task 0  (workload key: ["9847f8cc0b305137f4...) ==========
placeholder = PLACEHOLDER [1, 2048]
placeholder = PLACEHOLDER [1000, 2048]
T_dense(i, j) += (placeholder[i, k]*placeholder[j, k])
placeholder = PLACEHOLDER [1000]
T_add(ax0, ax1) = (T_dense[ax0, ax1] + placeholder[ax1])

*ret = SerializeFeaturePAM(features, fea_sizes, kmp_indexes, normalized_throughputs, task_ids, min_costs, byte_data)

features: 三维特征向量数组(N*buffer_seq*Buffer_Embedding_Dim)


pam_model.update
dataset.update_from_measure_pairs → get_per_store_features_from_measure_pairs_pam → load_task_data
fit_base → _fit_a_model → register_new_task → make_net



pytorch版本问题的解决
按照pruner的要求下载相应的pytorch，但是原来下载也是pruner要求的但是运行的时候在training model会报错，然后用下面的指令下载pruner指定的版本则不会报错
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html



best_costs[i]: 任务i的当前最佳延迟
task_cts[i]: 任务i被调优的次数
task_costs_history[i]: 任务i的历史延迟记录
flop_cts[i]: 任务i的浮点运算次数
alpha: 历史vs预期权重参数
beta: 相似任务性能差异阈值
FLOPS = flop_cts / best_costs
task_best_cts[i]: 记录任务i发现最佳结果的轮次

TaskScheduler的tune方法
梯度策略的核心： 在有限时间内，应该优先调优哪个任务，才能让任务整体性能提升最大
Chain Gradient(链式梯度): 评估该任务对整体目标的重要性
Backward Gradient(向后梯度): 评估该任务最近的表现
Forward Gradient(向前梯度): 评估该任务改进空间


task_tag: 任务标签
tag_to_group_id: 标签到group ID的映射
group_task_ids: 每个组包含的任务ID映射


segment_sizes_normal: 记录每个样本的特征段长度
flatten_normal_features: 存储所有普通特征
flatten_gemm_features: 存储GEMM相关的缓冲区特征


extractor(stmt) → 调用operator()
operator() → 调用VisitStmt()
VisitStmt() → 初始化vtable，调用vtable()
vtable() → 根据type_index查找函数
type_index() → 返回180(AttrStmtNode的ID)
func_[180] → 调用对应的lambda函数
lambda函数 → 类型转换并调用VisitStmt_
VisitStmt_(AttrStmtNode*) → 目标函数



te::InferBound
前馈图FeedGraph: 张量→消费者操作的映射
附加路径AttachPath: 操作→附加点的映射



PerStoreFeatureExtractorPAM类
ExtractBufferAccessFeature方法: 
cur_compute_ops: 当前计算操作数
compute_ops_list: 输出参数，各层循环的计算操作数列表
mem_bytes_list: 输出参数，各层循环的内存访问字节数列表
for_touch_regions_: 存储每个for循环的内存访问区域信息
buffer_regions_map: 当前循环层的buffer区域映射
tuple<BufferAccessTypePAM, int64_t, int>: 访问类型，访问元素数，单元素字节数


ComputeRegionPAM
ElementProduct
GetLoopExtentPAM
ComputeStridePAM
ComputeReusePAM


PAMDataLoader:

cmake -DCMAKE_BUILD_TYPE=Debug ..



Stmt部分代码生成逻辑
/src/te/schedule/schedule_ops.cc文件中的MakePipeline函数
生成producer代码→处理双缓冲优化→组合Producer和Consumer→添加内存管理→添加作用域标记(标记该operation的存储作用域)

python nltsp_train_single_gpu.py --dataset pkl_dataset/nltsp_dataset_t4_2213_4000_train.pkl --lr 0.00001 --epochs 50 --batch_size 2048 --weight_decay 1e-6 --num_workers 0 --model_name lstm --output_path


这是familyseer里面的时间消耗
Feature extraction scheduling time: 0.00138906 s
Feature extraction PrimFunc generation time: 0.00133408 s
Feature extraction GPU PrimFunc optimization time: 0.00326738 s
Feature extraction GPU PrimFunc simplify time: 0.000801187 s


[Timing] GetPerStoreFeaturesWorkerFuncPAM schedule preparation duration: 0.0299267 seconds
[Timing] GetPerStoreFeaturesWorkerFuncPAM TIR lowering duration: 0.0293026 seconds
[Timing] GetPerStoreFeaturesWorkerFuncPAM GPU simplify duration: 0.0192356 seconds


Epoch: 25       Batch: 2        Train Loss: 6.4319      Valid Loss: 0.0000      Train Speed: 74556      LR: 7.0000e-04
Epoch: 30       Batch: 2        Train Loss: 5.5979      Valid Loss: 0.0000      Train Speed: 75089      LR: 7.0000e-04
Epoch: 35       Batch: 2        Train Loss: 4.6660      Valid Loss: 0.0000      Train Speed: 73864      LR: 7.0000e-04
Epoch: 40       Batch: 2        Train Loss: 3.9200      Valid Loss: 0.0000      Train Speed: 75691      LR: 7.0000e-04
Epoch: 45       Batch: 2        Train Loss: 3.3814      Valid Loss: 0.0000      Train Speed: 73603      LR: 7.0000e-04
Epoch: 49       Batch: 2        Train Loss: 3.0914      Valid Loss: 0.0000      Train Speed: 73732      LR: 7.0000e-04
Time elapsed for training: 1.11 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.075 |        1895.78 |     64 |
|    3 |        0.073 |        1954.60 |     64 |
|    4 |        0.048 |        2418.39 |     64 |
|    5 |        0.045 |        2541.35 |     64 |
|    6 |        0.048 |        2384.08 |     64 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |     64 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |     64 |
|   12 |        0.027 |        4791.33 |     64 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.733 ms       Trials: 1148    Used time : 2453 s      Next ID: 2
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 106 fail_ct: 2966   Time elapsed: 9.50
population size #682
GA Iter: 0      Max score: 7.9665       Min score: 1.4323       #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 7.9665       Min score: 2.1087       #Pop: 128       #M+: 657        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 23.69
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................************************
Time elapsed for measurement: 87.64 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1212
Epoch: 0        Batch: 2        Train Loss: 47.2170     Valid Loss: 0.0000      Train Speed: 72011      LR: 7.0000e-04
Epoch: 5        Batch: 2        Train Loss: 28.6984     Valid Loss: 0.0000      Train Speed: 78799      LR: 7.0000e-04
Epoch: 10       Batch: 2        Train Loss: 18.4288     Valid Loss: 0.0000      Train Speed: 76118      LR: 7.0000e-04
Epoch: 15       Batch: 2        Train Loss: 12.8795     Valid Loss: 0.0000      Train Speed: 77379      LR: 7.0000e-04
Epoch: 20       Batch: 2        Train Loss: 9.6044      Valid Loss: 0.0000      Train Speed: 78933      LR: 7.0000e-04
Epoch: 25       Batch: 2        Train Loss: 7.7994      Valid Loss: 0.0000      Train Speed: 76455      LR: 7.0000e-04
Epoch: 30       Batch: 2        Train Loss: 6.4344      Valid Loss: 0.0000      Train Speed: 77612      LR: 7.0000e-04
Epoch: 35       Batch: 2        Train Loss: 5.5447      Valid Loss: 0.0000      Train Speed: 77389      LR: 7.0000e-04
Epoch: 40       Batch: 2        Train Loss: 4.5888      Valid Loss: 0.0000      Train Speed: 77683      LR: 7.0000e-04
Epoch: 45       Batch: 2        Train Loss: 3.8623      Valid Loss: 0.0000      Train Speed: 79452      LR: 7.0000e-04
Epoch: 49       Batch: 2        Train Loss: 3.9384      Valid Loss: 0.0000      Train Speed: 75142      LR: 7.0000e-04
Time elapsed for training: 1.60 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.067 |        2107.21 |    128 |
|    3 |        0.073 |        1954.60 |     64 |
|    4 |        0.048 |        2418.39 |     64 |
|    5 |        0.045 |        2541.35 |     64 |
|    6 |        0.048 |        2384.08 |     64 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |     64 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |     64 |
|   12 |        0.027 |        4791.33 |     64 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.718 ms       Trials: 1212    Used time : 2575 s      Next ID: 5
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 63  fail_ct: 1473   Time elapsed: 4.51
population size #639
GA Iter: 0      Max score: 6.7143       Min score: 1.5333       #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 6.7143       Min score: 1.6203       #Pop: 128       #M+: 659        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 23.01
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................************************
Time elapsed for measurement: 79.18 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1276
Epoch: 0        Batch: 2        Train Loss: 30.0443     Valid Loss: 0.0000      Train Speed: 73478      LR: 7.0000e-04
Epoch: 5        Batch: 2        Train Loss: 20.4762     Valid Loss: 0.0000      Train Speed: 80362      LR: 7.0000e-04
Epoch: 10       Batch: 2        Train Loss: 14.0835     Valid Loss: 0.0000      Train Speed: 80394      LR: 7.0000e-04
Epoch: 15       Batch: 2        Train Loss: 10.3320     Valid Loss: 0.0000      Train Speed: 79079      LR: 7.0000e-04
Epoch: 20       Batch: 2        Train Loss: 8.0076      Valid Loss: 0.0000      Train Speed: 81289      LR: 7.0000e-04
Epoch: 25       Batch: 2        Train Loss: 6.4663      Valid Loss: 0.0000      Train Speed: 81113      LR: 7.0000e-04
Epoch: 30       Batch: 2        Train Loss: 5.3676      Valid Loss: 0.0000      Train Speed: 82131      LR: 7.0000e-04
Epoch: 35       Batch: 2        Train Loss: 4.7543      Valid Loss: 0.0000      Train Speed: 82244      LR: 7.0000e-04
Epoch: 40       Batch: 2        Train Loss: 3.9336      Valid Loss: 0.0000      Train Speed: 77659      LR: 7.0000e-04
Epoch: 45       Batch: 2        Train Loss: 3.3320      Valid Loss: 0.0000      Train Speed: 82082      LR: 7.0000e-04
Epoch: 49       Batch: 2        Train Loss: 2.9635      Valid Loss: 0.0000      Train Speed: 78867      LR: 7.0000e-04
Time elapsed for training: 1.55 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.067 |        2107.21 |    128 |
|    3 |        0.073 |        1954.60 |     64 |
|    4 |        0.048 |        2418.39 |     64 |
|    5 |        0.045 |        2574.95 |    128 |
|    6 |        0.048 |        2384.08 |     64 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |     64 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |     64 |
|   12 |        0.027 |        4791.33 |     64 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.716 ms       Trials: 1276    Used time : 2684 s      Next ID: 3
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 55  fail_ct: 1481   Time elapsed: 4.75
population size #631
GA Iter: 0      Max score: 5.5637       Min score: 0.0982       #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 5.5637       Min score: 0.5250       #Pop: 128       #M+: 668        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 22.99
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................************************
Time elapsed for measurement: 83.05 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1340
Epoch: 0        Batch: 2        Train Loss: 54.1033     Valid Loss: 0.0000      Train Speed: 70001      LR: 7.0000e-04
Epoch: 5        Batch: 2        Train Loss: 32.9122     Valid Loss: 0.0000      Train Speed: 80919      LR: 7.0000e-04
Epoch: 10       Batch: 2        Train Loss: 20.9582     Valid Loss: 0.0000      Train Speed: 83655      LR: 7.0000e-04
Epoch: 15       Batch: 2        Train Loss: 14.5191     Valid Loss: 0.0000      Train Speed: 83573      LR: 7.0000e-04
Epoch: 20       Batch: 2        Train Loss: 10.7394     Valid Loss: 0.0000      Train Speed: 80369      LR: 7.0000e-04
Epoch: 25       Batch: 2        Train Loss: 8.9057      Valid Loss: 0.0000      Train Speed: 83646      LR: 7.0000e-04
Epoch: 30       Batch: 2        Train Loss: 7.1385      Valid Loss: 0.0000      Train Speed: 78043      LR: 7.0000e-04
Epoch: 35       Batch: 2        Train Loss: 5.8344      Valid Loss: 0.0000      Train Speed: 85028      LR: 7.0000e-04
Epoch: 40       Batch: 2        Train Loss: 4.6283      Valid Loss: 0.0000      Train Speed: 83383      LR: 7.0000e-04
Epoch: 45       Batch: 2        Train Loss: 3.7746      Valid Loss: 0.0000      Train Speed: 84869      LR: 7.0000e-04
Epoch: 49       Batch: 2        Train Loss: 3.4655      Valid Loss: 0.0000      Train Speed: 83288      LR: 7.0000e-04
Time elapsed for training: 1.52 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.067 |        2107.21 |    128 |
|    3 |        0.066 |        2169.05 |    128 |
|    4 |        0.048 |        2418.39 |     64 |
|    5 |        0.045 |        2574.95 |    128 |
|    6 |        0.048 |        2384.08 |     64 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |     64 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |     64 |
|   12 |        0.027 |        4791.33 |     64 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.709 ms       Trials: 1340    Used time : 2796 s      Next ID: 8
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 67  fail_ct: 1469   Time elapsed: 4.73
population size #643
GA Iter: 0      Max score: 4.6718       Min score: -0.0716      #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 4.6718       Min score: 0.0518       #Pop: 128       #M+: 655        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 27.23
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................************************
Time elapsed for measurement: 78.18 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1404
Epoch: 0        Batch: 2        Train Loss: 47.8604     Valid Loss: 0.0000      Train Speed: 76721      LR: 7.0000e-04
Epoch: 5        Batch: 2        Train Loss: 30.1485     Valid Loss: 0.0000      Train Speed: 86506      LR: 7.0000e-04
Epoch: 10       Batch: 2        Train Loss: 19.2889     Valid Loss: 0.0000      Train Speed: 85454      LR: 7.0000e-04
Epoch: 15       Batch: 2        Train Loss: 13.3028     Valid Loss: 0.0000      Train Speed: 86691      LR: 7.0000e-04
Epoch: 20       Batch: 2        Train Loss: 10.3571     Valid Loss: 0.0000      Train Speed: 84595      LR: 7.0000e-04
Epoch: 25       Batch: 2        Train Loss: 8.0546      Valid Loss: 0.0000      Train Speed: 86106      LR: 7.0000e-04
Epoch: 30       Batch: 2        Train Loss: 7.1820      Valid Loss: 0.0000      Train Speed: 85908      LR: 7.0000e-04
Epoch: 35       Batch: 2        Train Loss: 6.1032      Valid Loss: 0.0000      Train Speed: 88013      LR: 7.0000e-04
Epoch: 40       Batch: 2        Train Loss: 5.2370      Valid Loss: 0.0000      Train Speed: 74779      LR: 7.0000e-04
Epoch: 45       Batch: 2        Train Loss: 4.4783      Valid Loss: 0.0000      Train Speed: 85769      LR: 7.0000e-04
Epoch: 49       Batch: 2        Train Loss: 3.8160      Valid Loss: 0.0000      Train Speed: 83932      LR: 7.0000e-04
Time elapsed for training: 1.74 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.067 |        2107.21 |    128 |
|    3 |        0.066 |        2169.05 |    128 |
|    4 |        0.048 |        2418.39 |     64 |
|    5 |        0.045 |        2574.95 |    128 |
|    6 |        0.048 |        2384.08 |     64 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |    128 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |     64 |
|   12 |        0.027 |        4791.33 |     64 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.709 ms       Trials: 1404    Used time : 2909 s      Next ID: 2
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 58  fail_ct: 1478   Time elapsed: 4.59
====== Potential Space Generation =======
GA Iter: 0      Max score: 74.1306      Min score: 0.2023       #Pop: 58        #M+: 0  #M-: 0
GA Iter: 2      Max score: 80.6114      Min score: 36.3119      #Pop: 512       #M+: 874        #M-: 0
====== Potential Space Generation Time elapsed: 32.88 ======
population size #698
GA Iter: 0      Max score: 4.8583       Min score: 0.7039       #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 4.8583       Min score: 1.2972       #Pop: 128       #M+: 643        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 25.23
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................************************
Time elapsed for measurement: 83.40 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1468
Epoch: 0        Batch: 2        Train Loss: 34.4571     Valid Loss: 0.0000      Train Speed: 82124      LR: 7.0000e-04
Epoch: 5        Batch: 2        Train Loss: 23.6823     Valid Loss: 0.0000      Train Speed: 87684      LR: 7.0000e-04
Epoch: 10       Batch: 2        Train Loss: 16.8717     Valid Loss: 0.0000      Train Speed: 86870      LR: 7.0000e-04
Epoch: 15       Batch: 2        Train Loss: 12.8139     Valid Loss: 0.0000      Train Speed: 86560      LR: 7.0000e-04
Epoch: 20       Batch: 2        Train Loss: 9.8892      Valid Loss: 0.0000      Train Speed: 88070      LR: 7.0000e-04
Epoch: 25       Batch: 2        Train Loss: 8.3835      Valid Loss: 0.0000      Train Speed: 87775      LR: 7.0000e-04
Epoch: 30       Batch: 2        Train Loss: 7.0684      Valid Loss: 0.0000      Train Speed: 87000      LR: 7.0000e-04
Epoch: 35       Batch: 2        Train Loss: 5.9390      Valid Loss: 0.0000      Train Speed: 86159      LR: 7.0000e-04
Epoch: 40       Batch: 2        Train Loss: 5.0638      Valid Loss: 0.0000      Train Speed: 87596      LR: 7.0000e-04
Epoch: 45       Batch: 2        Train Loss: 4.3574      Valid Loss: 0.0000      Train Speed: 86168      LR: 7.0000e-04
Epoch: 49       Batch: 2        Train Loss: 3.8210      Valid Loss: 0.0000      Train Speed: 85189      LR: 7.0000e-04
Time elapsed for training: 1.70 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.067 |        2107.21 |    192 |
|    3 |        0.066 |        2169.05 |    128 |
|    4 |        0.048 |        2418.39 |     64 |
|    5 |        0.045 |        2574.95 |    128 |
|    6 |        0.048 |        2384.08 |     64 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |    128 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |     64 |
|   12 |        0.027 |        4791.33 |     64 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.709 ms       Trials: 1468    Used time : 3057 s      Next ID: 11
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 69  fail_ct: 1467   Time elapsed: 4.72
population size #645
GA Iter: 0      Max score: 8.4413       Min score: 0.7802       #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 8.4413       Min score: 1.5546       #Pop: 128       #M+: 640        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 27.21
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................********************************E********
........................************************
Time elapsed for measurement: 94.66 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1532
Epoch: 0        Batch: 2        Train Loss: 55.9508     Valid Loss: 0.0000      Train Speed: 83987      LR: 7.0000e-04
Epoch: 5        Batch: 2        Train Loss: 35.3849     Valid Loss: 0.0000      Train Speed: 90520      LR: 7.0000e-04
Epoch: 10       Batch: 2        Train Loss: 22.7989     Valid Loss: 0.0000      Train Speed: 89679      LR: 7.0000e-04
Epoch: 15       Batch: 2        Train Loss: 15.7621     Valid Loss: 0.0000      Train Speed: 89909      LR: 7.0000e-04
Epoch: 20       Batch: 2        Train Loss: 11.6395     Valid Loss: 0.0000      Train Speed: 90611      LR: 7.0000e-04
Epoch: 25       Batch: 2        Train Loss: 9.0670      Valid Loss: 0.0000      Train Speed: 90532      LR: 7.0000e-04
Epoch: 30       Batch: 2        Train Loss: 7.5595      Valid Loss: 0.0000      Train Speed: 90393      LR: 7.0000e-04
Epoch: 35       Batch: 2        Train Loss: 6.2645      Valid Loss: 0.0000      Train Speed: 90399      LR: 7.0000e-04
Epoch: 40       Batch: 2        Train Loss: 5.4346      Valid Loss: 0.0000      Train Speed: 88295      LR: 7.0000e-04
Epoch: 45       Batch: 2        Train Loss: 4.6396      Valid Loss: 0.0000      Train Speed: 88312      LR: 7.0000e-04
Epoch: 49       Batch: 2        Train Loss: 4.1413      Valid Loss: 0.0000      Train Speed: 89066      LR: 7.0000e-04
Time elapsed for training: 1.66 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.067 |        2107.21 |    192 |
|    3 |        0.066 |        2169.05 |    128 |
|    4 |        0.048 |        2418.39 |     64 |
|    5 |        0.045 |        2574.95 |    128 |
|    6 |        0.048 |        2384.08 |     64 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |    128 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |    128 |
|   12 |        0.027 |        4791.33 |     64 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.709 ms       Trials: 1532    Used time : 3185 s      Next ID: 12
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 66  fail_ct: 1470   Time elapsed: 4.77
population size #642
GA Iter: 0      Max score: 4.9452       Min score: -0.3728      #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 4.9452       Min score: 0.7258       #Pop: 128       #M+: 641        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 25.62
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................********************E****
Time elapsed for measurement: 95.65 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1596
Epoch: 0        Batch: 3        Train Loss: 61.1070     Valid Loss: 0.0000      Train Speed: 69757      LR: 7.0000e-04
Epoch: 5        Batch: 3        Train Loss: 29.9950     Valid Loss: 0.0000      Train Speed: 75573      LR: 7.0000e-04
Epoch: 10       Batch: 3        Train Loss: 16.5962     Valid Loss: 0.0000      Train Speed: 74764      LR: 7.0000e-04
Epoch: 15       Batch: 3        Train Loss: 10.5702     Valid Loss: 0.0000      Train Speed: 73363      LR: 7.0000e-04
Epoch: 20       Batch: 3        Train Loss: 7.6404      Valid Loss: 0.0000      Train Speed: 76791      LR: 7.0000e-04
Epoch: 25       Batch: 3        Train Loss: 5.8989      Valid Loss: 0.0000      Train Speed: 76309      LR: 7.0000e-04
Epoch: 30       Batch: 3        Train Loss: 5.5136      Valid Loss: 0.0000      Train Speed: 76385      LR: 7.0000e-04
Epoch: 35       Batch: 3        Train Loss: 4.3631      Valid Loss: 0.0000      Train Speed: 76392      LR: 7.0000e-04
Epoch: 40       Batch: 3        Train Loss: 3.6620      Valid Loss: 0.0000      Train Speed: 77412      LR: 7.0000e-04
Epoch: 45       Batch: 3        Train Loss: 3.3463      Valid Loss: 0.0000      Train Speed: 75999      LR: 7.0000e-04
Epoch: 49       Batch: 3        Train Loss: 2.9232      Valid Loss: 0.0000      Train Speed: 76102      LR: 7.0000e-04
Time elapsed for training: 1.89 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.067 |        2107.21 |    192 |
|    3 |        0.066 |        2169.05 |    128 |
|    4 |        0.048 |        2418.39 |     64 |
|    5 |        0.045 |        2574.95 |    128 |
|    6 |        0.048 |        2384.08 |     64 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |    128 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |    128 |
|   12 |        0.026 |        4940.72 |    128 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.708 ms       Trials: 1596    Used time : 3314 s      Next ID: 6
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 60  fail_ct: 1476   Time elapsed: 4.17
population size #636
GA Iter: 0      Max score: 5.9491       Min score: 0.3691       #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 5.9491       Min score: 1.4012       #Pop: 128       #M+: 661        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 24.05
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................************************
Time elapsed for measurement: 81.90 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1660
Epoch: 0        Batch: 3        Train Loss: 34.0673     Valid Loss: 0.0000      Train Speed: 74187      LR: 7.0000e-04
Epoch: 5        Batch: 3        Train Loss: 20.0955     Valid Loss: 0.0000      Train Speed: 77268      LR: 7.0000e-04
Epoch: 10       Batch: 3        Train Loss: 13.1691     Valid Loss: 0.0000      Train Speed: 77901      LR: 7.0000e-04
Epoch: 15       Batch: 3        Train Loss: 9.5755      Valid Loss: 0.0000      Train Speed: 78476      LR: 7.0000e-04
Epoch: 20       Batch: 3        Train Loss: 7.3863      Valid Loss: 0.0000      Train Speed: 78507      LR: 7.0000e-04
Epoch: 25       Batch: 3        Train Loss: 5.9216      Valid Loss: 0.0000      Train Speed: 80797      LR: 7.0000e-04
Epoch: 30       Batch: 3        Train Loss: 4.8602      Valid Loss: 0.0000      Train Speed: 74851      LR: 7.0000e-04
Epoch: 35       Batch: 3        Train Loss: 4.3220      Valid Loss: 0.0000      Train Speed: 81605      LR: 7.0000e-04
Epoch: 40       Batch: 3        Train Loss: 3.6073      Valid Loss: 0.0000      Train Speed: 77568      LR: 7.0000e-04
Epoch: 45       Batch: 3        Train Loss: 3.0761      Valid Loss: 0.0000      Train Speed: 79305      LR: 7.0000e-04
Epoch: 49       Batch: 3        Train Loss: 2.6045      Valid Loss: 0.0000      Train Speed: 79136      LR: 7.0000e-04
Time elapsed for training: 1.88 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.067 |        2107.21 |    192 |
|    3 |        0.066 |        2169.05 |    128 |
|    4 |        0.048 |        2418.39 |     64 |
|    5 |        0.045 |        2574.95 |    128 |
|    6 |        0.047 |        2428.18 |    128 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |    128 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |    128 |
|   12 |        0.026 |        4940.72 |    128 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.707 ms       Trials: 1660    Used time : 3426 s      Next ID: 4
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 59  fail_ct: 1477   Time elapsed: 1.91
population size #635
GA Iter: 0      Max score: 7.6809       Min score: -0.1834      #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 7.6809       Min score: 0.8037       #Pop: 128       #M+: 643        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 10.21
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................************************
Time elapsed for measurement: 76.52 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1724
Epoch: 0        Batch: 3        Train Loss: 32.0682     Valid Loss: 0.0000      Train Speed: 75331      LR: 7.0000e-04
Epoch: 5        Batch: 3        Train Loss: 20.6682     Valid Loss: 0.0000      Train Speed: 78448      LR: 7.0000e-04
Epoch: 10       Batch: 3        Train Loss: 13.4937     Valid Loss: 0.0000      Train Speed: 77405      LR: 7.0000e-04
Epoch: 15       Batch: 3        Train Loss: 9.1033      Valid Loss: 0.0000      Train Speed: 82537      LR: 7.0000e-04
Epoch: 20       Batch: 3        Train Loss: 6.7688      Valid Loss: 0.0000      Train Speed: 80998      LR: 7.0000e-04
Epoch: 25       Batch: 3        Train Loss: 5.4483      Valid Loss: 0.0000      Train Speed: 82823      LR: 7.0000e-04
Epoch: 30       Batch: 3        Train Loss: 5.1606      Valid Loss: 0.0000      Train Speed: 82933      LR: 7.0000e-04
Epoch: 35       Batch: 3        Train Loss: 4.9405      Valid Loss: 0.0000      Train Speed: 81095      LR: 7.0000e-04
Epoch: 40       Batch: 3        Train Loss: 4.1596      Valid Loss: 0.0000      Train Speed: 82194      LR: 7.0000e-04
Epoch: 45       Batch: 3        Train Loss: 3.4958      Valid Loss: 0.0000      Train Speed: 82811      LR: 7.0000e-04
Epoch: 49       Batch: 3        Train Loss: 2.9679      Valid Loss: 0.0000      Train Speed: 83159      LR: 7.0000e-04
Time elapsed for training: 1.50 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.067 |        2107.21 |    192 |
|    3 |        0.066 |        2169.05 |    128 |
|    4 |        0.048 |        2418.39 |    128 |
|    5 |        0.045 |        2574.95 |    128 |
|    6 |        0.047 |        2428.18 |    128 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |    128 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |    128 |
|   12 |        0.026 |        4940.72 |    128 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.707 ms       Trials: 1724    Used time : 3516 s      Next ID: 2
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 95  fail_ct: 2977   Time elapsed: 9.46
population size #760
GA Iter: 0      Max score: 2.2357       Min score: -1.6562      #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 2.2357       Min score: -0.4254      #Pop: 128       #M+: 650        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 26.06
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................***********************
Time elapsed for measurement: 85.54 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1788
Epoch: 0        Batch: 3        Train Loss: 39.1814     Valid Loss: 0.0000      Train Speed: 75150      LR: 7.0000e-04
Epoch: 5        Batch: 3        Train Loss: 22.6132     Valid Loss: 0.0000      Train Speed: 83008      LR: 7.0000e-04
Epoch: 10       Batch: 3        Train Loss: 14.5144     Valid Loss: 0.0000      Train Speed: 82424      LR: 7.0000e-04
Epoch: 15       Batch: 3        Train Loss: 10.7569     Valid Loss: 0.0000      Train Speed: 83447      LR: 7.0000e-04
Epoch: 20       Batch: 3        Train Loss: 8.7402      Valid Loss: 0.0000      Train Speed: 80820      LR: 7.0000e-04
Epoch: 25       Batch: 3        Train Loss: 6.7386      Valid Loss: 0.0000      Train Speed: 84131      LR: 7.0000e-04
Epoch: 30       Batch: 3        Train Loss: 5.3740      Valid Loss: 0.0000      Train Speed: 84297      LR: 7.0000e-04
Epoch: 35       Batch: 3        Train Loss: 4.3987      Valid Loss: 0.0000      Train Speed: 83992      LR: 7.0000e-04
Epoch: 40       Batch: 3        Train Loss: 3.7216      Valid Loss: 0.0000      Train Speed: 82991      LR: 7.0000e-04
Epoch: 45       Batch: 3        Train Loss: 3.4020      Valid Loss: 0.0000      Train Speed: 81114      LR: 7.0000e-04
Epoch: 49       Batch: 3        Train Loss: 2.9265      Valid Loss: 0.0000      Train Speed: 82769      LR: 7.0000e-04
Time elapsed for training: 1.89 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.057 |        2514.75 |    256 |
|    3 |        0.066 |        2169.05 |    128 |
|    4 |        0.048 |        2418.39 |    128 |
|    5 |        0.045 |        2574.95 |    128 |
|    6 |        0.047 |        2428.18 |    128 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |    128 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |    128 |
|   12 |        0.026 |        4940.72 |    128 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.685 ms       Trials: 1788    Used time : 3639 s      Next ID: 2
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 58  fail_ct: 1478   Time elapsed: 4.46
====== Potential Space Generation =======
GA Iter: 0      Max score: 71.0544      Min score: 0.0381       #Pop: 58        #M+: 0  #M-: 0
GA Iter: 2      Max score: 83.5835      Min score: 35.6872      #Pop: 512       #M+: 874        #M-: 0
====== Potential Space Generation Time elapsed: 34.09 ======
population size #723
GA Iter: 0      Max score: 8.2584       Min score: 1.1739       #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 8.2584       Min score: 2.4607       #Pop: 128       #M+: 651        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 24.66
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................************************
Time elapsed for measurement: 87.37 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1852
Epoch: 0        Batch: 3        Train Loss: 34.9243     Valid Loss: 0.0000      Train Speed: 79102      LR: 7.0000e-04
Epoch: 5        Batch: 3        Train Loss: 22.0637     Valid Loss: 0.0000      Train Speed: 83457      LR: 7.0000e-04
Epoch: 10       Batch: 3        Train Loss: 14.5229     Valid Loss: 0.0000      Train Speed: 83288      LR: 7.0000e-04
Epoch: 15       Batch: 3        Train Loss: 10.1470     Valid Loss: 0.0000      Train Speed: 84502      LR: 7.0000e-04
Epoch: 20       Batch: 3        Train Loss: 7.4818      Valid Loss: 0.0000      Train Speed: 84245      LR: 7.0000e-04
Epoch: 25       Batch: 3        Train Loss: 5.8192      Valid Loss: 0.0000      Train Speed: 83561      LR: 7.0000e-04
Epoch: 30       Batch: 3        Train Loss: 4.6474      Valid Loss: 0.0000      Train Speed: 83806      LR: 7.0000e-04
Epoch: 35       Batch: 3        Train Loss: 3.8080      Valid Loss: 0.0000      Train Speed: 84833      LR: 7.0000e-04
Epoch: 40       Batch: 3        Train Loss: 3.3210      Valid Loss: 0.0000      Train Speed: 82720      LR: 7.0000e-04
Epoch: 45       Batch: 3        Train Loss: 2.8848      Valid Loss: 0.0000      Train Speed: 84983      LR: 7.0000e-04
Epoch: 49       Batch: 3        Train Loss: 2.6064      Valid Loss: 0.0000      Train Speed: 85609      LR: 7.0000e-04
Time elapsed for training: 1.87 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.054 |        2648.64 |    320 |
|    3 |        0.066 |        2169.05 |    128 |
|    4 |        0.048 |        2418.39 |    128 |
|    5 |        0.045 |        2574.95 |    128 |
|    6 |        0.047 |        2428.18 |    128 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |    128 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |    128 |
|   12 |        0.026 |        4940.72 |    128 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.679 ms       Trials: 1852    Used time : 3792 s      Next ID: 5
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 55  fail_ct: 1481   Time elapsed: 4.27
====== Potential Space Generation =======
GA Iter: 0      Max score: 80.6307      Min score: 0.0524       #Pop: 55        #M+: 0  #M-: 0
GA Iter: 2      Max score: 85.0933      Min score: 47.1866      #Pop: 512       #M+: 874        #M-: 0
====== Potential Space Generation Time elapsed: 33.40 ======
population size #695
GA Iter: 0      Max score: 8.2583       Min score: 1.1018       #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 8.2583       Min score: 1.8254       #Pop: 128       #M+: 646        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 23.69
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................************************E
Time elapsed for measurement: 76.81 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1916
Epoch: 0        Batch: 3        Train Loss: 35.9319     Valid Loss: 0.0000      Train Speed: 79699      LR: 7.0000e-04
Epoch: 5        Batch: 3        Train Loss: 22.7918     Valid Loss: 0.0000      Train Speed: 85383      LR: 7.0000e-04
Epoch: 10       Batch: 3        Train Loss: 14.8491     Valid Loss: 0.0000      Train Speed: 86440      LR: 7.0000e-04
Epoch: 15       Batch: 3        Train Loss: 10.7238     Valid Loss: 0.0000      Train Speed: 85004      LR: 7.0000e-04
Epoch: 20       Batch: 3        Train Loss: 8.3768      Valid Loss: 0.0000      Train Speed: 86076      LR: 7.0000e-04
Epoch: 25       Batch: 3        Train Loss: 6.4905      Valid Loss: 0.0000      Train Speed: 85978      LR: 7.0000e-04
Epoch: 30       Batch: 3        Train Loss: 5.1914      Valid Loss: 0.0000      Train Speed: 87109      LR: 7.0000e-04
Epoch: 35       Batch: 3        Train Loss: 4.2973      Valid Loss: 0.0000      Train Speed: 87162      LR: 7.0000e-04
Epoch: 40       Batch: 3        Train Loss: 3.7243      Valid Loss: 0.0000      Train Speed: 87639      LR: 7.0000e-04
Epoch: 45       Batch: 3        Train Loss: 3.0762      Valid Loss: 0.0000      Train Speed: 84144      LR: 7.0000e-04
Epoch: 49       Batch: 3        Train Loss: 2.6472      Valid Loss: 0.0000      Train Speed: 87591      LR: 7.0000e-04
Time elapsed for training: 1.86 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.054 |        2648.64 |    320 |
|    3 |        0.066 |        2169.05 |    128 |
|    4 |        0.048 |        2418.39 |    128 |
|    5 |        0.039 |        2948.54 |    192 |
|    6 |        0.047 |        2428.18 |    128 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |    128 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |    128 |
|   12 |        0.026 |        4940.72 |    128 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.668 ms       Trials: 1916    Used time : 3932 s      Next ID: 8
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 64  fail_ct: 1472   Time elapsed: 4.75
====== Potential Space Generation =======
GA Iter: 0      Max score: 81.2065      Min score: 0.0090       #Pop: 64        #M+: 0  #M-: 0
GA Iter: 2      Max score: 120.5216     Min score: 58.4200      #Pop: 512       #M+: 893        #M-: 0
====== Potential Space Generation Time elapsed: 37.85 ======
population size #704
GA Iter: 0      Max score: 5.9917       Min score: -2.0006      #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 5.9917       Min score: 0.4408       #Pop: 128       #M+: 643        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 27.44
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................************************
Time elapsed for measurement: 75.61 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 1980
Epoch: 0        Batch: 3        Train Loss: 44.5713     Valid Loss: 0.0000      Train Speed: 81389      LR: 7.0000e-04
Epoch: 5        Batch: 3        Train Loss: 26.9889     Valid Loss: 0.0000      Train Speed: 80866      LR: 7.0000e-04
Epoch: 10       Batch: 3        Train Loss: 17.7034     Valid Loss: 0.0000      Train Speed: 87245      LR: 7.0000e-04
Epoch: 15       Batch: 3        Train Loss: 13.9445     Valid Loss: 0.0000      Train Speed: 88954      LR: 7.0000e-04
Epoch: 20       Batch: 3        Train Loss: 10.1880     Valid Loss: 0.0000      Train Speed: 85809      LR: 7.0000e-04
Epoch: 25       Batch: 3        Train Loss: 7.5026      Valid Loss: 0.0000      Train Speed: 89736      LR: 7.0000e-04
Epoch: 30       Batch: 3        Train Loss: 5.8173      Valid Loss: 0.0000      Train Speed: 89414      LR: 7.0000e-04
Epoch: 35       Batch: 3        Train Loss: 4.5843      Valid Loss: 0.0000      Train Speed: 88797      LR: 7.0000e-04
Epoch: 40       Batch: 3        Train Loss: 4.1158      Valid Loss: 0.0000      Train Speed: 89560      LR: 7.0000e-04
Epoch: 45       Batch: 3        Train Loss: 3.5026      Valid Loss: 0.0000      Train Speed: 89416      LR: 7.0000e-04
Epoch: 49       Batch: 3        Train Loss: 3.0285      Valid Loss: 0.0000      Train Speed: 60728      LR: 7.0000e-04
Time elapsed for training: 2.01 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.054 |        2648.64 |    320 |
|    3 |        0.066 |        2169.05 |    128 |
|    4 |        0.048 |        2418.39 |    128 |
|    5 |        0.039 |        2948.54 |    192 |
|    6 |        0.047 |        2428.18 |    128 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |    192 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |    128 |
|   12 |        0.026 |        4940.72 |    128 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.668 ms       Trials: 1980    Used time : 4081 s      Next ID: 3
----------------------------------------------------------------------
------------------------------  [ Search ]
----------------------------------------------------------------------
Sample Initial Population       #s: 94  fail_ct: 2978   Time elapsed: 9.20
====== Potential Space Generation =======
GA Iter: 0      Max score: 59.4959      Min score: 0.0550       #Pop: 94        #M+: 0  #M-: 0
GA Iter: 2      Max score: 80.6205      Min score: 37.0808      #Pop: 512       #M+: 872        #M-: 0
====== Potential Space Generation Time elapsed: 34.42 ======
population size #734
GA Iter: 0      Max score: 7.2730       Min score: 0.9450       #Pop: 128       #M+: 0  #M-: 0
GA Iter: 1      Max score: 7.2730       Min score: 2.0071       #Pop: 128       #M+: 659        #M-: 0
EvolutionarySearch      #s: 128 Time elapsed: 25.09
----------------------------------------------------------------------
------------------------------  [ Measure ]
----------------------------------------------------------------------
Get 64 programs to measure:
........................................****************************************
........................************************
Time elapsed for measurement: 86.07 s
----------------------------------------------------------------------
------------------------------  [ Train cost model ]
----------------------------------------------------------------------
============================================================
Fit a net. Train size: 2044
Epoch: 0        Batch: 3        Train Loss: 42.2754     Valid Loss: 0.0000      Train Speed: 81799      LR: 7.0000e-04
Epoch: 5        Batch: 3        Train Loss: 25.8361     Valid Loss: 0.0000      Train Speed: 88407      LR: 7.0000e-04
Epoch: 10       Batch: 3        Train Loss: 16.8245     Valid Loss: 0.0000      Train Speed: 88991      LR: 7.0000e-04
Epoch: 15       Batch: 3        Train Loss: 12.2315     Valid Loss: 0.0000      Train Speed: 79775      LR: 7.0000e-04
Epoch: 20       Batch: 3        Train Loss: 9.2761      Valid Loss: 0.0000      Train Speed: 89156      LR: 7.0000e-04
Epoch: 25       Batch: 3        Train Loss: 7.3847      Valid Loss: 0.0000      Train Speed: 88443      LR: 7.0000e-04
Epoch: 30       Batch: 3        Train Loss: 6.4856      Valid Loss: 0.0000      Train Speed: 87530      LR: 7.0000e-04
Epoch: 35       Batch: 3        Train Loss: 5.4452      Valid Loss: 0.0000      Train Speed: 88886      LR: 7.0000e-04
Epoch: 40       Batch: 3        Train Loss: 4.4460      Valid Loss: 0.0000      Train Speed: 88936      LR: 7.0000e-04
Epoch: 45       Batch: 3        Train Loss: 3.7733      Valid Loss: 0.0000      Train Speed: 88922      LR: 7.0000e-04
Epoch: 49       Batch: 3        Train Loss: 3.4538      Valid Loss: 0.0000      Train Speed: 90682      LR: 7.0000e-04
Time elapsed for training: 2.00 s
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.007 |         140.65 |     64 |
|    1 |        0.003 |          -0.00 |     64 |
|    2 |        0.054 |        2648.64 |    320 |
|    3 |        0.052 |        2733.32 |    192 |
|    4 |        0.048 |        2418.39 |    128 |
|    5 |        0.039 |        2948.54 |    192 |
|    6 |        0.047 |        2428.18 |    128 |
|    7 |        0.028 |        4138.69 |     64 |
|    8 |        0.036 |        3536.24 |    192 |
|    9 |        0.030 |        4172.04 |     64 |
|   10 |        0.019 |        6124.79 |     64 |
|   11 |        0.031 |        4128.14 |    128 |
|   12 |        0.026 |        4940.72 |    128 |
|   13 |        0.005 |         384.06 |     64 |
|   14 |        0.025 |        9393.75 |     64 |
|   15 |        0.004 |        2894.39 |     64 |
|   16 |        0.005 |        2501.15 |     64 |
|   17 |        0.008 |        1669.34 |     64 |
-------------------------------------------------
Estimated total latency: 0.654 ms       Trials: 2044    Used time : 4238 s      Next ID: -1
[04:32:37] /home/zk/Pruner/src/runtime/threading_backend.cc:217: warning: more than two frequencies detected!




安装transformer==3.5.0步骤
conda activate pruner

# 用 conda 安装可以避免编译问题
conda install -c conda-forge sentencepiece
pip install transformers==3.5.0 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tokenizers==0.9.4 -i https://pypi.tuna.tsinghua.edu.cn/simple







