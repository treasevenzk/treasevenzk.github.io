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

### CPU部分的实验

python run.py -p dlboost -c quick_start.json

程序运行经过历程:<br>
HeronRunner.LocalRunner &rarr; autotvm.measure_option &rarr; op_name("gemm"),case([64, 64, 64]) &rarr; 

config配置的属性: <br>
target_name、verbose、measure_time_per_round、use_cost_model、out_name、parallel、parallel_num
pop_num、iter_walks、history_topk
feature_type、loss_type
select_num、crossover_key_ratio
temperature、feasible_file_path
opt_method、max_trials、runner_number、runner_repeat、runner_timeout、build_timeout、in_dtype、out_dtype、cases、temperature <br>
device_id、tuned、codegen_type、get_op

Env类 <br>
config: 
runner: Runner类
num_ops: len(all_opt_methods)
build_kwargs:
task:
tuner:
pre_buffer:

Task类 <br>
name:   、 args:    、 opfunc:   、 target:     、 target_host:     、  build_kwargs:      、 knob_manager:     、 not_verify_correctness:      、 ref_input:      、
ref_output:     、 config:

knobManager类 <br>
sched_tups:     、 is_building:     、 solver:      、 axis_parents:        、 axis_brother:       、 axis_ori_lenth:计算并存储轴的长度、 mems:        、 staged_fused_axes:标记已融合的数据并行轴
knob_names:     、 solved_knob_vals_genotype:       、 solved_knob_vals_phenotype:      、 candidates:      、 _valid:      、 constraint_descs:       、 dump_descs:

Tuner类 <br>
config:     、 iter_no:     、 cost_model:      、 total_sample_time:       、 total_measure_time:      、 cost_model:      

perfBuffer类 <br>
perfs:      、 data_x:      、 data_y:      、 samples:     、 measured_keys:       、 best_perf:       、 best_sample:     、 config:

Context类 <br>
sched_desc:     、 codegen_type:        、 target_name:     、 scheduled_axes:      、 build_kwargs:       、 pos_via_tag:      、 tensor_dict:将张量存储到字典中，使用张量名称作为键、 input_tensors:整个计算图的输入张量    
axis_anotations:        、 stage_orgnize:       、 no_schedule_stages:     、  inlined_stages:     、vectorized_stages:      、 unrolled_stages:      、 general_tile_stages:
tensorized_stages:      、 tiled_stages:      、 stile_structures:       、 rtile_structures:       、 unroll_pragma_desc:      、 compute_poses:记录已被融合的计算阶段的信息、 compute_pos_names:
tensorize_info:     、 knob_manager:

CPUContext类 <br>
parallel_stages:       、 cached_stages:        、 unpack_info:        、 codegen_type:         、 tensorize_info:      、 stage_organize:      


