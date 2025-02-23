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
HeronRunner.LocalRunner &rarr; autotvm.measure_option &rarr; op_name("gemm"),case([64, 64, 64]) &rarr; Env &rarr; run &rarr; env.createTask
&rarr; Task &rarr; self.tuner &rarr; self.tuner.buildCostModel &rarr; self.perf_buffer &rarr; env.tune &rarr; self.task.make_stage_schedules() &rarr;
{buildContext &rarr; CPUContext &rarr; heron_dense &rarr; tvm.te.create_schedule &rarr; init_tensor_dict &rarr; sched_via_rule &rarr; ctx.updateAxisLength
getStageNamesOrdered &rarr; ctx.addSched &rarr; self.knob_manager.sched_tups.append &rarr; get_op_methods &rarr; op_methods &rarr; action.perform
computeAtOp.perform &rarr; compute_at &rarr; self.genAxesLength &rarr; fixAxesLength &rarr; genFuseAxisLength &rarr; printAxes &rarr; (mergeConsumer、unrollPragma、parallel、tileForCache、tensorize、generalTile、vectorize) &rarr; self.dump_constraints()} &rarr; {self.runner.measure_batch &rarr; self.dump_schedule() &rarr; self.tuner.run &rarr; self.check_feasible_exits &rarr;
self.constrained_random_sample &rarr; constrained_random_sample_parallel &rarr; Job(constrained_random_sample_sequential, (task, work_load, config), timeout) &rarr; job.start &rarr; job.get &rarr;
self.predict &rarr; self.UpdatePopulation &rarr; self.constrained_random_sample &rarr; self.history_topk_samples &rarr; self.repredict &rarr; self.optimize &rarr;
self.epsilon_select &rarr; self.FilterSamples &rarr; self.RouletteWheelSelection &rarr; self.FilterSamples &rarr; self.measure} &rarr; task.apply_best &rarr; 


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
sched_tups: 调度方法+stage_name    、 is_building:     、 solver:      、 axis_parents:循环分块，原循环和分割后循环之间的关系、 axis_brother:       、 axis_ori_lenth:记录每个轴的信息 算子名+变量名+轴长度 mems:        、 staged_fused_axes: 记录cache_write的算子的轴的信息 算子名+变量名
knob_names:调优规则(类似P#ST:dense,AX:None,PA:global_pos)、 solved_knob_vals_genotype:       、 solved_knob_vals_phenotype:      、 candidates:      、 _valid:      、 constraint_descs:       、 dump_descs:

Tuner类 <br>
config:     、 iter_no:     、 cost_model:      、 total_sample_time:       、 total_measure_time:      、 cost_model:      

perfBuffer类 <br>
perfs:      、 data_x:      、 data_y:      、 samples:     、 measured_keys:       、 best_perf:       、 best_sample:     、 config:

Context类 <br>
sched_desc:     、 codegen_type:        、 target_name:     、 scheduled_axes:记录循环分块的轴、 build_kwargs:       、 pos_via_tag:      、 tensor_dict:将张量存储到字典中，使用张量名称作为键、 input_tensors:整个计算图的输入张量    
axis_anotations:记录循环分块的信息      、 stage_orgnize:       、 no_schedule_stages:     、  inlined_stages:     、vectorized_stages:      、 unrolled_stages:记录循环分块的阶段信息、 general_tile_stages:
tensorized_stages:      、 tiled_stages:      、 stile_structures:记录循环分块的结构信息、 rtile_structures:       、 unroll_pragma_desc: 记录循环分块的信息     、 compute_poses:记录cache存放的位置、 compute_pos_names:记录cache存放位置的名字
tensorize_info:     、 knob_manager:

CPUContext类 <br>
parallel_stages:       、 cached_stages:采用cache_write的算子 、 unpack_info:        、 codegen_type:         、 tensorize_info:      、 stage_organize:      

schedOp类 <br>


Job类 <br>
func:       、 attach_info:         、 timeout:

Sample类 <br>
valid:      、 perf:        、 task:        、 knob_manager:        、 predict:         、 prob:        、 violation:       、 violations:         、 ranks:


