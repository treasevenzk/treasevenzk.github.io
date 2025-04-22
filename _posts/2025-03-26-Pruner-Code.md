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
88+1521+174+1698+576+483+477+176+1880
1764+975+

search_policy
126+124+1242+790+503