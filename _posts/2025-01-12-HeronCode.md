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
valid: 求解是否可行、 perf:        、 task:        、 knob_manager:深拷贝复制原来的knob_manager、 predict:         、 prob:        、 violation:       、 violations:         、 ranks:         、point:求解的值(推测的)
stmat_code: 求解的值换一种形式表示


上面的信息记录如下：
==== finish sched_via_rule ====
ctx.sched_desc: 
## Cache write global
dense_global = s.cache_write(dense, global)

#==--------- Start schedule STAGE dense ----------==#

## Unroll pragma 
i_o, i_i = s[dense].split(i, nparts = 1)
j_o, j_i = s[dense].split(j, nparts = 1)
s[dense].reorder(i_o, j_o, i_i, j_i, )

## Parallel 

## tile spatial 
i_i_o, i_i_i = s[dense].split(i_i, nparts = 1)
j_i_o, j_i_i = s[dense].split(j_i, nparts = 1)
s[dense].reorder(i_i_o, j_i_o, i_i_i, j_i_i, )
i_i_o_j_i_o_f = s[dense].fuse(i_i_o, j_i_o, )
s[dense].parallel(i_i_o_j_i_o_f)

## Tile for cache 

## tile spatial 
i_i_i_o, i_i_i_i = s[dense].split(i_i_i, nparts = 1)
j_i_i_o, j_i_i_i = s[dense].split(j_i_i, nparts = 1)
s[dense].reorder(i_i_i_o, j_i_i_o, i_i_i_i, j_i_i_i, )

# Var i_o length 1
# Var j_o length 1
# Var i_i_o_j_i_o_f length 1
# Var i_i_i_o length 1
# Var j_i_i_o length 1
# Var i_i_i_i length 1
# Var j_i_i_i length 1
#==--------- Start schedule STAGE dense.global ----------==#
s[dense_global].compute_at(s[dense], j_o)

# Var i_c length 1
# Var j_c length 1
# Var k
## general tile 

## tile 
i_c_o, i_c_i = s[dense_global].split(i_c, nparts = 1)
j_c_o, j_c_i = s[dense_global].split(j_c, nparts = 1)
k_o, k_i = s[dense_global].split(k, nparts = 1)
s[dense_global].reorder(i_c_o, j_c_o, k_o, i_c_i, j_c_i, k_i, )

## tile 
i_c_i_o, i_c_i_i = s[dense_global].split(i_c_i, nparts = 1)
j_c_i_o, j_c_i_i = s[dense_global].split(j_c_i, nparts = 1)
k_i_o, k_i_i = s[dense_global].split(k_i, nparts = 1)
s[dense_global].reorder(i_c_i_o, j_c_i_o, k_i_o, i_c_i_i, j_c_i_i, k_i_i, )

## tile 
i_c_i_i_o, i_c_i_i_i = s[dense_global].split(i_c_i_i, nparts = 1)
j_c_i_i_o, j_c_i_i_i = s[dense_global].split(j_c_i_i, nparts = 1)
k_i_i_o, k_i_i_i = s[dense_global].split(k_i_i, nparts = 1)
s[dense_global].reorder(i_c_i_i_o, j_c_i_i_o, k_i_i_o, i_c_i_i_i, j_c_i_i_i, k_i_i_i, )

# Var i_c_o length 1
# Var j_c_o length 1
# Var k_o length 1
# Var i_c_i_o length 1
# Var j_c_i_o length 1
# Var k_i_o length 1
# Var i_c_i_i_o length 1
# Var j_c_i_i_o length 1
# Var k_i_i_o length 1
# Var i_c_i_i_i length 1
# Var j_c_i_i_i length 1
# Var k_i_i_i length 1
#==--------- Start schedule STAGE B ----------==#

#==--------- Start schedule STAGE A ----------==#

ctx.scheduled_axes: ['dense_i.outer', 'dense_j.outer', 'dense_i.inner.outer', 'dense_j.inner.outer', 'dense_i.inner.outer.j.inner.outer.fused', 'dense_i.inner.inner.outer', 'dense_j.inner.inner.outer', 'dense.global_i.c.outer', 'dense.global_j.c.outer', 'dense.global_k.outer', 'dense.global_i.c.inner.outer', 'dense.global_j.c.inner.outer', 'dense.global_k.inner.outer', 'dense.global_i.c.inner.inner.outer', 'dense.global_j.c.inner.inner.outer', 'dense.global_k.inner.inner.outer']
ctx.axis_anotations: {'L#ST:dense,AX:i.outer': 'unroll', 'L#ST:dense,AX:j.outer': 'unroll'}
ctx.stile_structres: {'dense': [(['i.outer', 'j.outer'], 'unroll'), (['i.inner.outer', 'j.inner.outer'], 'None'), (['i.inner.inner.outer', 'j.inner.inner.outer'], 'None')], 'dense.global': [(['i.c.outer', 'j.c.outer'], 'None'), (['i.c.inner.outer', 'j.c.inner.outer'], 'None'), (['i.c.inner.inner.outer', 'j.c.inner.inner.outer'], 'None')]}
ctx.unroll_pragma_desc: {'dense': ('i.outer', 'dense_unroll_pragma')}
ctx.compute_poses: {'dense.global': ('dense', 'dense_global_pos')}
ctx.compute_pos_names: {'dense': ['j.outer', 'i.inner.outer.j.inner.outer.fused', 'j.inner.inner.outer', 'j.inner.inner.inner']}
ctx.knob_manager.axis_parents: {'L#ST:dense,AX:i.outer': ['L#ST:dense,AX:i'], 'L#ST:dense,AX:i.inner': ['L#ST:dense,AX:i'], 'L#ST:dense,AX:j.outer': ['L#ST:dense,AX:j'], 'L#ST:dense,AX:j.inner': ['L#ST:dense,AX:j'], 'L#ST:dense,AX:i.inner.outer': ['L#ST:dense,AX:i.inner'], 'L#ST:dense,AX:i.inner.inner': ['L#ST:dense,AX:i.inner'], 'L#ST:dense,AX:j.inner.outer': ['L#ST:dense,AX:j.inner'], 'L#ST:dense,AX:j.inner.inner': ['L#ST:dense,AX:j.inner'], 'L#ST:dense,AX:i.inner.outer.j.inner.outer.fused': ['L#ST:dense,AX:i.inner.outer', 'L#ST:dense,AX:j.inner.outer'], 'L#ST:dense,AX:i.inner.inner.outer': ['L#ST:dense,AX:i.inner.inner'], 'L#ST:dense,AX:i.inner.inner.inner': ['L#ST:dense,AX:i.inner.inner'], 'L#ST:dense,AX:j.inner.inner.outer': ['L#ST:dense,AX:j.inner.inner'], 'L#ST:dense,AX:j.inner.inner.inner': ['L#ST:dense,AX:j.inner.inner'], 'L#ST:dense.global,AX:i.c.outer': ['L#ST:dense.global,AX:i.c'], 'L#ST:dense.global,AX:i.c.inner': ['L#ST:dense.global,AX:i.c'], 'L#ST:dense.global,AX:j.c.outer': ['L#ST:dense.global,AX:j.c'], 'L#ST:dense.global,AX:j.c.inner': ['L#ST:dense.global,AX:j.c'], 'L#ST:dense.global,AX:k.outer': ['L#ST:dense.global,AX:k'], 'L#ST:dense.global,AX:k.inner': ['L#ST:dense.global,AX:k'], 'L#ST:dense.global,AX:i.c.inner.outer': ['L#ST:dense.global,AX:i.c.inner'], 'L#ST:dense.global,AX:i.c.inner.inner': ['L#ST:dense.global,AX:i.c.inner'], 'L#ST:dense.global,AX:j.c.inner.outer': ['L#ST:dense.global,AX:j.c.inner'], 'L#ST:dense.global,AX:j.c.inner.inner': ['L#ST:dense.global,AX:j.c.inner'], 'L#ST:dense.global,AX:k.inner.outer': ['L#ST:dense.global,AX:k.inner'], 'L#ST:dense.global,AX:k.inner.inner': ['L#ST:dense.global,AX:k.inner'], 'L#ST:dense.global,AX:i.c.inner.inner.outer': ['L#ST:dense.global,AX:i.c.inner.inner'], 'L#ST:dense.global,AX:i.c.inner.inner.inner': ['L#ST:dense.global,AX:i.c.inner.inner'], 'L#ST:dense.global,AX:j.c.inner.inner.outer': ['L#ST:dense.global,AX:j.c.inner.inner'], 'L#ST:dense.global,AX:j.c.inner.inner.inner': ['L#ST:dense.global,AX:j.c.inner.inner'], 'L#ST:dense.global,AX:k.inner.inner.outer': ['L#ST:dense.global,AX:k.inner.inner'], 'L#ST:dense.global,AX:k.inner.inner.inner': ['L#ST:dense.global,AX:k.inner.inner']}
ctx.knob_manager.axis_brother: {'L#ST:dense,AX:i.inner': 'L#ST:dense,AX:i.outer', 'L#ST:dense,AX:i.outer': 'L#ST:dense,AX:i.inner', 'L#ST:dense,AX:j.inner': 'L#ST:dense,AX:j.outer', 'L#ST:dense,AX:j.outer': 'L#ST:dense,AX:j.inner', 'L#ST:dense,AX:i.inner.inner': 'L#ST:dense,AX:i.inner.outer', 'L#ST:dense,AX:i.inner.outer': 'L#ST:dense,AX:i.inner.inner', 'L#ST:dense,AX:j.inner.inner': 'L#ST:dense,AX:j.inner.outer', 'L#ST:dense,AX:j.inner.outer': 'L#ST:dense,AX:j.inner.inner', 'L#ST:dense,AX:i.inner.inner.inner': 'L#ST:dense,AX:i.inner.inner.outer', 'L#ST:dense,AX:i.inner.inner.outer': 'L#ST:dense,AX:i.inner.inner.inner', 'L#ST:dense,AX:j.inner.inner.inner': 'L#ST:dense,AX:j.inner.inner.outer', 'L#ST:dense,AX:j.inner.inner.outer': 'L#ST:dense,AX:j.inner.inner.inner', 'L#ST:dense.global,AX:i.c.inner': 'L#ST:dense.global,AX:i.c.outer', 'L#ST:dense.global,AX:i.c.outer': 'L#ST:dense.global,AX:i.c.inner', 'L#ST:dense.global,AX:j.c.inner': 'L#ST:dense.global,AX:j.c.outer', 'L#ST:dense.global,AX:j.c.outer': 'L#ST:dense.global,AX:j.c.inner', 'L#ST:dense.global,AX:k.inner': 'L#ST:dense.global,AX:k.outer', 'L#ST:dense.global,AX:k.outer': 'L#ST:dense.global,AX:k.inner', 'L#ST:dense.global,AX:i.c.inner.inner': 'L#ST:dense.global,AX:i.c.inner.outer', 'L#ST:dense.global,AX:i.c.inner.outer': 'L#ST:dense.global,AX:i.c.inner.inner', 'L#ST:dense.global,AX:j.c.inner.inner': 'L#ST:dense.global,AX:j.c.inner.outer', 'L#ST:dense.global,AX:j.c.inner.outer': 'L#ST:dense.global,AX:j.c.inner.inner', 'L#ST:dense.global,AX:k.inner.inner': 'L#ST:dense.global,AX:k.inner.outer', 'L#ST:dense.global,AX:k.inner.outer': 'L#ST:dense.global,AX:k.inner.inner', 'L#ST:dense.global,AX:i.c.inner.inner.inner': 'L#ST:dense.global,AX:i.c.inner.inner.outer', 'L#ST:dense.global,AX:i.c.inner.inner.outer': 'L#ST:dense.global,AX:i.c.inner.inner.inner', 'L#ST:dense.global,AX:j.c.inner.inner.inner': 'L#ST:dense.global,AX:j.c.inner.inner.outer', 'L#ST:dense.global,AX:j.c.inner.inner.outer': 'L#ST:dense.global,AX:j.c.inner.inner.inner', 'L#ST:dense.global,AX:k.inner.inner.inner': 'L#ST:dense.global,AX:k.inner.inner.outer', 'L#ST:dense.global,AX:k.inner.inner.outer': 'L#ST:dense.global,AX:k.inner.inner.inner'}
ctx.knob_manager.axis_ori_lenth: {'L#ST:dense,AX:i': 64, 'L#ST:dense,AX:j': 64, 'L#ST:dense,AX:k': 64, 'L#ST:dense.global,AX:i.c': 64, 'L#ST:dense.global,AX:j.c': 64, 'L#ST:dense.global,AX:k': 64}
ctx.knob_manager.knob_names: ['dense_global_pos', 'dense_unroll_pragma', 'dense_tileSpatial', 'dense_vectorize', 'dense.global_tileAll']
ctx.knob_manager.candidates: {'dense_unroll_pragma': [0, 1, 2, 3, 4, 5], 'dense_tileSpatial': [1, 2, 4, 8, 16, 32, 64], 'dense.global_tileAll': [1, 2, 4, 8, 16, 32, 64]}
ctx.knob_manager.solver.vals: {'dense_global_pos': Var(dense_global_pos, 0, 3, 0), 'dense_unroll_pragma': Var(dense_unroll_pragma, 0, 5, 0), 'dense_i': Var(dense_i, 1, 64, 1), 'dense_i.outer': Var(dense_i.outer, 1, 64, 1), 'dense_i.inner': Var(dense_i.inner, 1, 64, 1), 'dense_j': Var(dense_j, 1, 64, 1), 'dense_j.outer': Var(dense_j.outer, 1, 64, 1), 'dense_j.inner': Var(dense_j.inner, 1, 64, 1), 'dense_tileSpatial': Var(dense_tileSpatial, 1, 64, 1), 'dense_i.inner.outer': Var(dense_i.inner.outer, 1, 64, 1), 'dense_i.inner.inner': Var(dense_i.inner.inner, 1, 64, 1), 'dense_j.inner.outer': Var(dense_j.inner.outer, 1, 64, 1), 'dense_j.inner.inner': Var(dense_j.inner.inner, 1, 64, 1), 'dense_i.inner.outer.j.inner.outer.fused': Var(dense_i.inner.outer.j.inner.outer.fused, 1, 1000000, 1), 'dense_i.inner.inner.outer': Var(dense_i.inner.inner.outer, 1, 64, 1), 'dense_i.inner.inner.inner': Var(dense_i.inner.inner.inner, 1, 64, 1), 'dense_j.inner.inner.outer': Var(dense_j.inner.inner.outer, 1, 64, 1), 'dense_j.inner.inner.inner': Var(dense_j.inner.inner.inner, 1, 64, 1), 'dense_vectorize': Var(dense_vectorize, 0, 1, 0), 'dense.global_i.c': Var(dense.global_i.c, 1, 64, 1), 'dense_global_pos_select0': Var(dense_global_pos_select0, 0, 1, 0), 'dense_global_pos_select1': Var(dense_global_pos_select1, 0, 1, 0), 'dense_global_pos_select2': Var(dense_global_pos_select2, 0, 1, 0), 'dense_global_pos_select3': Var(dense_global_pos_select3, 0, 1, 0), 'dense.global_j.c': Var(dense.global_j.c, 1, 64, 1), 'dense.global_tileAll': Var(dense.global_tileAll, 1, 64, 1), 'dense.global_i.c.outer': Var(dense.global_i.c.outer, 1, 64, 1), 'dense.global_i.c.inner': Var(dense.global_i.c.inner, 1, 64, 1), 'dense.global_j.c.outer': Var(dense.global_j.c.outer, 1, 64, 1), 'dense.global_j.c.inner': Var(dense.global_j.c.inner, 1, 64, 1), 'dense.global_k': Var(dense.global_k, 1, 64, 1), 'dense.global_k.outer': Var(dense.global_k.outer, 1, 64, 1), 'dense.global_k.inner': Var(dense.global_k.inner, 1, 64, 1), 'dense.global_i.c.inner.outer': Var(dense.global_i.c.inner.outer, 1, 64, 1), 'dense.global_i.c.inner.inner': Var(dense.global_i.c.inner.inner, 1, 64, 1), 'dense.global_j.c.inner.outer': Var(dense.global_j.c.inner.outer, 1, 64, 1), 'dense.global_j.c.inner.inner': Var(dense.global_j.c.inner.inner, 1, 64, 1), 'dense.global_k.inner.outer': Var(dense.global_k.inner.outer, 1, 64, 1), 'dense.global_k.inner.inner': Var(dense.global_k.inner.inner, 1, 64, 1), 'dense.global_i.c.inner.inner.outer': Var(dense.global_i.c.inner.inner.outer, 1, 64, 1), 'dense.global_i.c.inner.inner.inner': Var(dense.global_i.c.inner.inner.inner, 1, 64, 1), 'dense.global_j.c.inner.inner.outer': Var(dense.global_j.c.inner.inner.outer, 1, 64, 1), 'dense.global_j.c.inner.inner.inner': Var(dense.global_j.c.inner.inner.inner, 1, 64, 1), 'dense.global_k.inner.inner.outer': Var(dense.global_k.inner.inner.outer, 1, 64, 1), 'dense.global_k.inner.inner.inner': Var(dense.global_k.inner.inner.inner, 1, 64, 1)}


print("==== finish sched_via_rule ====")
print(f"ctx.sched_desc: {ctx.sched_desc}")
print(f"ctx.scheduled_axes: {ctx.scheduled_axes}")
print(f"ctx.axis_anotations: {ctx.axis_anotations}")
print(f"ctx.stile_structres: {ctx.stile_structures}")
print(f"ctx.unroll_pragma_desc: {ctx.unroll_pragma_desc}")
print(f"ctx.compute_poses: {ctx.compute_poses}")
print(f"ctx.compute_pos_names: {ctx.compute_pos_names}")
print(f"ctx.knob_manager.axis_parents: {ctx.knob_manager.axis_parents}")
print(f"ctx.knob_manager.axis_brother: {ctx.knob_manager.axis_brother}")
print(f"ctx.knob_manager.axis_ori_lenth: {ctx.knob_manager.axis_ori_lenth}")
print(f"ctx.knob_manager.knob_names: {ctx.knob_manager.knob_names}")
print(f"ctx.knob_manager.candidates: {ctx.knob_manager.candidates}")
print(f"ctx.knob_manager.solver.vals: {ctx.knob_manager.solver.vals}")
print("===============================")


调优过程：








TVM中的内容:
from tvm.autotvm.measure.measure import MeasureInput: MeasureInput类在TVM的AutoTVM模块中的作用是封装测量特定张量操作配置性能所需的信息；存储任务(要优化的张量操作)和要测量的特定配置、包含测量基础设施编译和运行操作特定实现所需的信息、作为输入提供给实际基准测试不同配置性能的测量模块；有助于为特定硬件目标找到张量操作的最佳实现方案




Heron整理流程

前期准备工作
1. 关于build_kwargs这一块的内容可以学习一下，利用claude举例说明
2. 关于张量化操作应该怎么用，仔细调研一下
3. 调研一下在利用cache_write原语之后，s会发生什么变化，重点关注一下轴、算子的变化
4. python语法知识，就是返回两个列表，但是我接收的时候只有一个列表，用claude举例说明
5. 将代码丢给claude，然后让claude帮忙写create_measure_batch、MeasureInput的例子

调度原语(14个)
split、reorder、parallel、fuse、bind、unrollPragma、vectorize、cache_read、cache_write、compute_at、compute_inline、set_scope、tensorize、storage_algin

cache_write:  ctx.tensor_dict
compute_at:
vectorize: ctx.vectorized_stages、ctx.axis_anotations
split: ctx.knob_manager.axis_parents (ctx.compute_pos_names)  ctx.scheduled_axes  添加约束条件ProdTwo、EQ knob_manager.axis_brother/constraint_descs
reorder:
fuse: ctx.knob_manager.axis_parents  ctx.scheduled_axes (ctx.compute_pos_names)  ctx 添加约束条件EQ、ProdTwo  knob_manager.constraint_descs
parallel:
tensorize: ctx.tensorized_stages   ctx.axis_anotations
vectorize: ctx.scheduled_axes   ctx.vectorized_stages      ctx.axis_anotations
compute_at:


调度原语:
addCacheWriteGlobalOp:   cache_write      ctx.cached_stages、ctx.compute_poses、ctx.compute_pos_names
startOp:
mergeConsumerOp: compute_at、vectorize
unrollPragmaOp: split     ctx.knob_manager.candidates   ctx.axis_anotations  ctx.stile_structures  ctx.unroll_pragma_desc  ctx.unrolled_staged
parallelOp->TileSpatialOp->TileAllOp->schedOp:  split、reorder、fuse、parallel   ctx.knob_manager.candidates ctx.stile_structures ctx.parallel_stages
tileForCacheOp->TileSpatialOp->TileAllOp->schedOp: split、reorder   ctx.knob_manager.candidate ctx.stile_structures
tensorizOp: split、reorder、tensorize       ctx.stile_structure   ctx.rtile_structure
generalTileOp->TileAllop->schedOp: split、reorder  ctx.knob_manager.candidates   ctx.stile_structure ctx  ctx.rtile_structure
CPUvectorizeOp: vectorize
CPUfinishOp->finishOp: 
startOp->computeAtOp: compute_at  添加约束条件EQ、ProdTwo/Sum

Generation
space generator: schedule generation rules(schedule template generation)、constraint generation rules(constrain generation)
调度模板生成：在代码中利用7条规则(正好和论文中对应)
do_merge_consumer:  compute_at、vectorize
do_unroll: isRootStage这条规则制定的原因
do_parallel: isRootStage
do_tile_for_cache:
do_tensorize:
do_generaltile: hasDataReuse
do_vectorize:

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



Exploration
space explorer--->measure--->cost model


---
### GPU部分的实验

print("prepare work")
print(f"ctx.default_sharedload_stages: {ctx.default_sharedload_stages}")
print(f"ctx.shared_load_stages: {ctx.shared_load_stages}")
print(f"ctx.no_schedule_stages: {ctx.no_schedule_stages}")
print(f"s.outputs: {s.outputs}")
for stage in s.stages:
    print(f"stage: {stage}")



addCacheTensorCoreOp类中发生的改变
ctx.shared_load_stages          ctx.default_shareload_stages        ctx.tensorize_loadA_stage       ctx.tensorize_loadB_stage
ctx.tensorize_com_stage         ctx.tensor_store_stage              ctx.pos_via_tag

准备过程：

ctx.default_sharedload_stages: ['A.shared', 'B.shared']
ctx.shared_load_stages: ['A.shared', 'B.shared', 'dense.wmma.accumulator.shared']
ctx.no_schedule_stages: []
ctx.compute_poses: {'A.shared': ('dense.wmma.accumulator', 'dense.wmma.accumulator_shared_pos'), 'B.shared': ('dense.wmma.accumulator', 'dense.wmma.accumulator_shared_pos'), 'A.shared.wmma.matrix_a': ('dense.wmma.accumulator', 'dense.wmma.accumulator_local_pos'), 'B.shared.wmma.matrix_b': ('dense.wmma.accumulator', 'dense.wmma.accumulator_local_pos'), 'dense.wmma.accumulator.shared': ('dense', 'dense_shared_pos'), 'dense.wmma.accumulator': ('dense.wmma.accumulator.shared', 'dense.wmma.accumulator.shared_local_pos')}
ctx.knob_manager.solver.vals: {'wmma_m': Var(wmma_m, 8, 32, 16), 'wmma_m_cand0': Var(wmma_m_cand0, 0, 1, 0), 'wmma_m_cand1': Var(wmma_m_cand1, 0, 1, 0), 'wmma_m_cand2': Var(wmma_m_cand2, 0, 1, 0), 'wmma_k': Var(wmma_k, 16, 16, 16), 'wmma_n': Var(wmma_n, 8, 32, 16), 'wmma_n_cand0': Var(wmma_n_cand0, 0, 1, 0), 'wmma_n_cand1': Var(wmma_n_cand1, 0, 1, 0), 'wmma_n_cand2': Var(wmma_n_cand2, 0, 1, 0), 'wmma_m_wmma_n': Var(wmma_m_wmma_n, 1, 4096, 1), 'dense.wmma.accumulator_shared_pos': Var(dense.wmma.accumulator_shared_pos, 0, 10, 0), 'dense.wmma.accumulator_local_pos': Var(dense.wmma.accumulator_local_pos, 0, 10, 0), 'dense_shared_pos': Var(dense_shared_pos, 0, 10, 0), 'dense.wmma.accumulator.shared_local_pos': Var(dense.wmma.accumulator.shared_local_pos, 0, 10, 0)}
s.outputs: [compute(dense, body=[T.reduce(T.comm_reducer(lambda x, y: x + y, [T.float16(0)]), source=[A[i, k] * B[j, k]], init=[], axis=[T.iter_var(k, T.Range(0, 64), "CommReduce", "")], condition=T.bool(True), value_index=0)], axis=[T.iter_var(i, T.Range(0, 64), "DataPar", ""), T.iter_var(j, T.Range(0, 64), "DataPar", "")], reduce_axis=[T.iter_var(k, T.Range(0, 64), "CommReduce", "")], tag=batch_matmul, attrs={})]
stage: stage(A, placeholder(A, 0x344dffe0))
stage: stage(A.shared, compute(A.shared, body=[A[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 64), "DataPar", ""), T.iter_var(ax1, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=, attrs={}))
stage: stage(A.shared.wmma.matrix_a, compute(A.shared.wmma.matrix_a, body=[A.shared[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 64), "DataPar", ""), T.iter_var(ax1, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=, attrs={}))
stage: stage(B, placeholder(B, 0x3419af40))
stage: stage(B.shared, compute(B.shared, body=[B[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 64), "DataPar", ""), T.iter_var(ax1, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=, attrs={}))
stage: stage(B.shared.wmma.matrix_b, compute(B.shared.wmma.matrix_b, body=[B.shared[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 64), "DataPar", ""), T.iter_var(ax1, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=, attrs={}))
stage: stage(dense.wmma.accumulator, compute(dense.wmma.accumulator, body=[T.reduce(T.comm_reducer(lambda x, y: x + y, [T.float16(0)]), source=[A.shared.wmma.matrix_a[i_c, k] * B.shared.wmma.matrix_b[j_c, k]], init=[], axis=[T.iter_var(k, T.Range(0, 64), "CommReduce", "")], condition=T.bool(True), value_index=0)], axis=[T.iter_var(i_c, T.Range(0, 64), "DataPar", ""), T.iter_var(j_c, T.Range(0, 64), "DataPar", "")], reduce_axis=[T.iter_var(k, T.Range(0, 64), "CommReduce", "")], tag=batch_matmul, attrs={}))
stage: stage(dense.wmma.accumulator.shared, compute(dense.wmma.accumulator.shared, body=[dense.wmma.accumulator[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 64), "DataPar", ""), T.iter_var(ax1, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=, attrs={}))
stage: stage(dense, compute(dense, body=[dense.wmma.accumulator.shared[i, j]], axis=[T.iter_var(i, T.Range(0, 64), "DataPar", ""), T.iter_var(j, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=batch_matmul, attrs={}))


addCacheTensorCoreOp: cache_write、cache_read*2、cache_read*2、cache_read
ctx.shared_load_stages: A.shared、B.shared、dense.wmma.accumulator.shared
ctx.default_sharedload_stages: A.shared、B.shared
ctx.compute_poses: A.shared、B.shared、A.shared.wmma.matrix_a、B.shared.wmma.matrix_b、dense.wmma.accumulator.shared、dense.wmma.accumulator
ctx.tensorize_com_stage: dense.wmma.accumulator
ctx.tensorize_store_stage: dense.wmma.accumulator.shared
ctx.tensorize_loadA_stage: A.shared.wmma.matrix_a
ctx.tensorize_loadB_stage: B.shared.wmma.matrix_b

dense
TCStartOp->startOp:

unrollPragmaOp: split、reorder
ctx.unrolled_stages: dense

tileBlockOp->tileBindOp->TileSpatialOp:split、reorder、fuse、bind
ctx.bind_block_stages: dense

tileThreadOp->tileBindOp->TileSpatialOp: split、reorder、fuse、bind
ctx.stage_warp_nums: dense
ctx.bind_thread_stages: dense

tileWarpOp->tileBindOp->TIleSpatialOp: split、reorder、fuse、bind
ctx.bind_warp_stages: dense

GPUvectorizeOp->TileSpatialOp: split、reorder、fuse、vectorize


split里面关于dense_i是否要的问题，get_ax(ax_key) 在unrollPragma作者提供的样例是64


ctx.scheduled_axes: split、fuse、bind、unrollPragma、vectorize
ctx.compute_pos_names: split、fuse
ctx.unrolled_stages: unrollPragma
ctx.vectorized_stages: vectorize
ctx.axis_anotations: unrollPragma、vectorize、tensorize、tensorize_x86、tensorize_vta
ctx.double_buffered_stages: double_buffer
ctx.tensor_dict: cache_read、cache_write
ctx.no_schedule_stages: compute_inline
ctx.tensorized_stages: tensorize、tensorize_x86、tensorize_vta


addCacheTensorCoreOp->addCacheReadShareOp: ctx.knob_manager.candidates、ctx.tensorize_loadA_stage、ctx.tensorize_loadB_stage、ctx.tensorize_com_size、ctx.tensorize_store_size、ctx.pos_via_tag、ctx.shared_load_stages、ctx.default_sharedload_stages、ctx.compute_pos_names
TCStartOp->startOp->computeAtOp:
defaultSharedLoadSchedOp->storagemAlignOp: ctx.align_sizes、ctx.bind_warp_stages、ctx.bind_thread_stages
defaultSchedOp: ctx.knob_manager.addCandidates、ctx.bind_thread_stages
unrollPragmaOp: ctx.knob_manager.candidates、ctx.axis_anotations、ctx.stile_structures、ctx.unroll_pragma_desc、ctx.unrolled_stages
storageAlignOp: ctx.align_sizes
tileBlockOp->tileBindOp->TileSpatialOp: ctx.knob_manager.candidates、ctx.stile_structures、ctx.bind_block_stages
tileThreadOp->tileBindOp->TileSpatialOp: ctx.knob_manager.candidates、ctx.stile_structures、ctx.bind_thread_stages
tileWarpOp->tilebindOp->TileSpatialOp: ctx.konb_manager.candidates、ctx.stile_structures、ctx.bind_warp_stages
generalTileOp->TileAllOp: ctx.knob_manager.candidates、ctx.stile_structures、ctx.rtile_structures
tensorcoreLoadAOp: ctx.stile_structures
tensorcoreLoadBOp: ctx.stile_structures
tensorcoreComputeOp: ctx.stile_structures、ctx.rtile_structures
tensorcoreStoreOp: ctx.stile_structures
GPUvectorizeOp->TileSpatialOp: ctx.knob_manager.candidates、ctx.stile_structures
GPUfinishOp->finishOp: 


blockId.x : block_pos_idx = 1
threadId.x : thread_pos_idx = 3
threadidx.y : warp_pos_idx = None
unroll: unroll_pos_idx = 0

addCacheTensorCoreOp->addCacheReadShareOp: 
ctx.knob_manager.candidates、ctx.tensorize_loadA_stage、ctx.tensorize_loadB_stage、ctx.tensorize_com_size、ctx.tensorize_store_size、ctx.pos_via_tag、ctx.shared_load_stages、ctx.default_sharedload_stages、ctx.compute_poses、ctx.compute_pos_names(这个在后续split和fuse，会改变每个stage里面轴分割和融合的记录信息)

dense
TCStartOp->startOp:
unrollPragmaOp: ctx.knob_manager.candidates、ctx.axis_anotations、ctx.stile_structures、ctx.unroll_pragma_desc、ctx.unrolled_stages
tileBlockOp->tileBindOp->TileSpatialOp: ctx.knob_manager.candidates、ctx.stile_structures、ctx.bind_block_stages
tileThreadOp->tileBindOp->TileSpatialOp: ctx.knob_manager.candidates、ctx.stile_structures、ctx.bind_thread_stages
tileWarpOp->tilebindOp->TileSpatialOp: ctx.konb_manager.candidates、ctx.stile_structures、ctx.bind_warp_stages
GPUvectorizeOp->TileSpatialOp: ctx.knob_manager.candidates、ctx.stile_structures
GPUfinishOp->finishOp: 

dense.wmma.accumulator.shared
TCStartOp->startOp->computeAtOp:
storageAlignOp: ctx.align_sizes
tileThreadOp->tileBindOp->TileSpatialOp: ctx.knob_manager.candidates、ctx.stile_structures、ctx.bind_thread_stages
tensorcoreStoreOp: ctx.stile_structures



TCStartOp→startOp→computeAtOp
defaultSharedLoadSchedOp→storageAlignOp→fuseAllOp
defaultSchedOp→fuseAllOp
unrollPragmaOp
storageAlignOp
tileBlockOp→tileBindOp→TileSpatialOp
tileThreadOp→tileBindOp→TileSpatialOp
tileWarpOp→tileBindOp→TileSpatialOp
generalTileOp→TileAllOp
tensorcoreLoadAOp
tensorcoreLoadBOp
tensorcoreComputeOp
tensorcoreStoreOp
GPUvectorizeOp→tileSpatialOp

tuner.run

UpdatePopulation->optimize->epsilon_select->FilterSamples->measure

UpdatePopulation
constrained_random_sample->history_topk_samples

constrained_random_sample
constrainted_random_sample_parallel->self.predict

constrainted_random_sample_sequential
Sample、knob_manager.randomSample、Code



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



# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((64, 64), "float16"), B: T.Buffer((64, 64), "float16"), dense: T.Buffer((64, 64), "float16")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
        dense_wmma_accumulator_shared = T.allocate([4096], "float16", "shared")
        with T.launch_thread("threadIdx.y", 1) as threadIdx_y:
            dense_wmma_accumulator = T.allocate([4096], "float16", "wmma.accumulator")
            A_shared = T.allocate([4096], "float16", "shared")
            A_shared_wmma_matrix_a = T.allocate([4096], "float16", "wmma.matrix_a")
            B_shared_wmma_matrix_b = T.allocate([4096], "float16", "wmma.matrix_b")
            for i_c_inner_inner_inner_outer_init, j_c_inner_inner_inner_outer_init in T.grid(4, 4):
                T.tvm_fill_fragment(dense_wmma_accumulator, 16, 16, 16, i_c_inner_inner_inner_outer_init * 4 + j_c_inner_inner_inner_outer_init, T.float32(0))
            for ax0_ax1_fused_outer_outer_outer in range(4096):
                threadIdx_y_1 = T.launch_thread("threadIdx.y", 1)
                threadIdx_x = T.launch_thread("threadIdx.x", 1)
                A_shared_1 = T.Buffer((4096,), "float16", data=A_shared, scope="shared")
                A_1 = T.Buffer((4096,), "float16", data=A.data)
                A_shared_1[ax0_ax1_fused_outer_outer_outer] = A_1[ax0_ax1_fused_outer_outer_outer]
            for ax0_outer, ax1_outer in T.grid(4, 4):
                T.tvm_load_matrix_sync(A_shared_wmma_matrix_a, 16, 16, 16, ax0_outer * 4 + ax1_outer, T.tvm_access_ptr(T.type_annotation("float16"), A_shared, ax0_outer * 1024 + ax1_outer * 16, 1024, 1), 64, "row_major")
            for ax0_ax1_fused_outer_outer_outer in range(4096):
                threadIdx_y_1 = T.launch_thread("threadIdx.y", 1)
                threadIdx_x = T.launch_thread("threadIdx.x", 1)
                A_shared_1 = T.Buffer((4096,), "float16", data=A_shared, scope="shared")
                B_1 = T.Buffer((4096,), "float16", data=B.data)
                A_shared_1[ax0_ax1_fused_outer_outer_outer] = B_1[ax0_ax1_fused_outer_outer_outer]
            for ax0_outer, ax1_outer in T.grid(4, 4):
                T.tvm_load_matrix_sync(B_shared_wmma_matrix_b, 16, 16, 16, ax0_outer * 4 + ax1_outer, T.tvm_access_ptr(T.type_annotation("float16"), A_shared, ax0_outer * 1024 + ax1_outer * 16, 1024, 1), 64, "col_major")
            for i_c_inner_inner_inner_outer, j_c_inner_inner_inner_outer, k_inner_inner_inner_outer in T.grid(4, 4, 4):
                cse_var_2: T.int32 = i_c_inner_inner_inner_outer * 4
                cse_var_1: T.int32 = cse_var_2 + j_c_inner_inner_inner_outer
                T.tvm_mma_sync(dense_wmma_accumulator, cse_var_1, A_shared_wmma_matrix_a, cse_var_2 + k_inner_inner_inner_outer, B_shared_wmma_matrix_b, j_c_inner_inner_inner_outer * 4 + k_inner_inner_inner_outer, dense_wmma_accumulator, cse_var_1)
            for ax0_inner_outer, ax1_inner_outer in T.grid(4, 4):
                T.tvm_store_matrix_sync(dense_wmma_accumulator, 16, 16, 16, ax0_inner_outer * 4 + ax1_inner_outer, T.tvm_access_ptr(T.type_annotation("float16"), dense_wmma_accumulator_shared, ax0_inner_outer * 1024 + ax1_inner_outer * 16, 1024, 2), 64, "row_major")
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_y = T.launch_thread("threadIdx.y", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 1)
        dense_1 = T.Buffer((4096,), "float16", data=dense.data)
        dense_wmma_accumulator_shared_1 = T.Buffer((4096,), "float16", data=dense_wmma_accumulator_shared, scope="shared")
        dense_1[0:4096] = dense_wmma_accumulator_shared_1[0:4096]




# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((64, 64), "float16"), B: T.Buffer((64, 64), "float16"), dense: T.Buffer((64, 64), "float16")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
        blockIdx_x = T.launch_thread("blockIdx.x", 4)
        dense_wmma_accumulator = T.allocate([512], "float16", "wmma.accumulator")
        A_shared = T.allocate([512], "float16", "shared")
        B_shared = T.allocate([2048], "float16", "shared")
        A_shared_wmma_matrix_a = T.allocate([256], "float16", "wmma.matrix_a")
        B_shared_wmma_matrix_b = T.allocate([256], "float16", "wmma.matrix_b")
        threadIdx_y = T.launch_thread("threadIdx.y", 2)
        with T.launch_thread("threadIdx.y", 2) as threadIdx_y_1:
            for j_c_inner_inner_inner_outer_init in T.unroll(2):
                T.tvm_fill_fragment(dense_wmma_accumulator, 16, 16, 16, j_c_inner_inner_inner_outer_init, T.float32(0))
            for k_outer in T.unroll(4):
                cse_var_1: T.int32 = k_outer * 16
                with T.launch_thread("threadIdx.y", 2) as threadIdx_y_2:
                    threadIdx_x = T.launch_thread("threadIdx.x", 32)
                    A_shared_1 = T.Buffer((512,), "float16", data=A_shared, scope="shared")
                    A_1 = T.Buffer((4096,), "float16", data=A.data)
                    A_shared_1[threadIdx_y_2 * 256 + threadIdx_x * 8:threadIdx_y_2 * 256 + threadIdx_x * 8 + 8] = A_1[blockIdx_x // 2 * 2048 + threadIdx_y_2 * 1024 + threadIdx_x // 2 * 64 + cse_var_1 + threadIdx_x % 2 * 8:blockIdx_x // 2 * 2048 + threadIdx_y_2 * 1024 + threadIdx_x // 2 * 64 + cse_var_1 + threadIdx_x % 2 * 8 + 8]
                with T.launch_thread("threadIdx.y", 2) as threadIdx_y_2:
                    threadIdx_x = T.launch_thread("threadIdx.x", 32)
                    B_shared_1 = T.Buffer((1536,), "float16", data=B_shared, scope="shared")
                    B_1 = T.Buffer((4096,), "float16", data=B.data)
                    B_shared_1[threadIdx_y_2 * 768 + threadIdx_x // 2 * 48 + threadIdx_x % 2 * 8:threadIdx_y_2 * 768 + threadIdx_x // 2 * 48 + threadIdx_x % 2 * 8 + 8] = B_1[blockIdx_x % 2 * 2048 + threadIdx_y_2 * 1024 + threadIdx_x // 2 * 64 + cse_var_1 + threadIdx_x % 2 * 8:blockIdx_x % 2 * 2048 + threadIdx_y_2 * 1024 + threadIdx_x // 2 * 64 + cse_var_1 + threadIdx_x % 2 * 8 + 8]
                for j_c_inner_inner_inner_outer in T.unroll(2):
                    T.tvm_load_matrix_sync(A_shared_wmma_matrix_a, 16, 16, 16, 0, T.tvm_access_ptr(T.type_annotation("float16"), A_shared, threadIdx_y_1 * 256, 256, 1), 16, "row_major")
                    T.tvm_load_matrix_sync(B_shared_wmma_matrix_b, 16, 16, 16, 0, T.tvm_access_ptr(T.type_annotation("float16"), B_shared, j_c_inner_inner_inner_outer * 768, 768, 1), 48, "col_major")
                    T.tvm_mma_sync(dense_wmma_accumulator, j_c_inner_inner_inner_outer, A_shared_wmma_matrix_a, 0, B_shared_wmma_matrix_b, 0, dense_wmma_accumulator, j_c_inner_inner_inner_outer)
            for ax1_inner_outer in T.unroll(2):
                T.tvm_store_matrix_sync(dense_wmma_accumulator, 16, 16, 16, ax1_inner_outer, T.tvm_access_ptr(T.type_annotation("float16"), B_shared, threadIdx_y_1 * 1024 + ax1_inner_outer * 16, 1024, 2), 64, "row_major")
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        for i_inner_inner_inner_inner_outer in T.unroll(4):
            for j_inner_inner_inner_inner_outer in T.unroll(2):
                cse_var_3: T.int32 = i_inner_inner_inner_inner_outer * 64
                cse_var_2: T.int32 = j_inner_inner_inner_inner_outer * 2
                dense_1 = T.Buffer((4096,), "float16", data=dense.data)
                B_shared_1 = T.Buffer((2048,), "float16", data=B_shared, scope="shared")
                dense_1[blockIdx_x // 2 * 2048 + threadIdx_x // 4 * 256 + cse_var_3 + blockIdx_x % 2 * 32 + threadIdx_y * 16 + threadIdx_x % 4 * 4 + cse_var_2:blockIdx_x // 2 * 2048 + threadIdx_x // 4 * 256 + cse_var_3 + blockIdx_x % 2 * 32 + threadIdx_y * 16 + threadIdx_x % 4 * 4 + cse_var_2 + 2] = B_shared_1[threadIdx_x // 4 * 256 + cse_var_3 + threadIdx_y * 16 + threadIdx_x % 4 * 4 + cse_var_2:threadIdx_x // 4 * 256 + cse_var_3 + threadIdx_y * 16 + threadIdx_x % 4 * 4 + cse_var_2 + 2]



修改Heron过程

sched_cand:

{'wmma_m_cand0': 1, 'wmma_m_cand1': 1, 'wmma_m_cand2': 1, 'wmma_n_cand0': 1, 'wmma_n_cand1': 1, 'wmma_n_cand2': 1, 'wmma_m_wmma_n': 1, 'dense_vectorize_cand0': 1, 'dense_vectorize_cand1': 1, 'dense_vectorize_cand2': 1, 'dense_vectorize_cand3': 1, 'dense_shared_pos_select0': 1, 'dense_shared_pos_select1': 1, 'dense_shared_pos_select2': 1, 'dense_shared_pos_select3': 1, 'dense_shared_pos_select4': 1, 'dense_shared_pos_select5': 1, 'dense.wmma.accumulator.shared_offset_cand0': 1, 'dense.wmma.accumulator.shared_offset_cand1': 1, 'dense.wmma.accumulator.shared_offset_cand2': 1, 'dense.wmma.accumulator.shared_offset_cand3': 1, 'dense.wmma.accumulator.shared_offset_cand4': 1, 'dense.wmma.accumulator.shared_offset_cand5': 1, 'dense.wmma.accumulator.shared_local_pos_select0': 1, 'dense.wmma.accumulator.shared_local_pos_select1': 1, 'dense.wmma.accumulator_shared_pos_select0': 1, 'dense.wmma.accumulator_shared_pos_select1': 1, 'dense.wmma.accumulator_shared_pos_select2': 1, 'dense.wmma.accumulator_shared_pos_select3': 1, 'B.shared_offset_cand0': 1, 'B.shared_offset_cand1': 1, 'B.shared_offset_cand2': 1, 'B.shared_offset_cand3': 1, 'B.shared_offset_cand4': 1, 'B.shared_offset_cand5': 1, 'B.shared_vectorize_cand0': 1, 'B.shared_vectorize_cand1': 1, 'B.shared_vectorize_cand2': 1, 'B.shared_vectorize_cand3': 1, 'dense.wmma.accumulator_local_pos_select0': 1, 'dense.wmma.accumulator_local_pos_select1': 1, 'dense.wmma.accumulator_local_pos_select2': 1, 'dense.wmma.accumulator_local_pos_select3': 1, 'A.shared_offset_cand0': 1, 'A.shared_offset_cand1': 1, 'A.shared_offset_cand2': 1, 'A.shared_offset_cand3': 1, 'A.shared_offset_cand4': 1, 'A.shared_offset_cand5': 1, 'A.shared_vectorize_cand0': 1, 'A.shared_vectorize_cand1': 1, 'A.shared_vectorize_cand2': 1, 'A.shared_vectorize_cand3': 1}



{'dense.wmma.accumulator.shared_offset_cand5', 'dense_j.inner.inner', 'dense.wmma.accumulator.shared_shared_mem_size', 'dense.wmma.accumulator_k.inner.inner', 'dense_shared_pos_select5', 'dense_vectorize_cand3', 'dense_vectorize_cand0', 'wmma_n_cand0', 'dense_shared_pos_select1', 'dense.wmma.accumulator_local_pos_select2', 'wmma_m_cand2', 'dense.wmma.accumulator.shared_offset_cand0', 'dense.wmma.accumulator_shared_pos_select3', 'dense_i.inner', 'dense.wmma.accumulator_j.c.inner.inner', 'dense.wmma.accumulator_shared_pos_select2', 'B.shared_ax0.ax1.fused', 'dense_j.inner.inner.outer', 'dense_i.inner.outer', 'B.shared_offset_cand1', 'A.shared_align_size', 'dense.wmma.accumulator.shared_local_pos_select1', 'B.shared_shared_mem_size', 'dense_vectorize_cand1', 'dense_j.inner', 'dense_i.inner.inner.inner.inner.inner', 'B.shared_offset_cand0', 'A.shared_offset_cand4', 'dense.wmma.accumulator.shared_ax1.inner', 'A.shared_vectorize_cand0', 'B.shared_vectorize_cand1', 'dense_j.inner.inner.inner.inner', 'A.shared_shared_mem_size', 'A.shared_ax0.ax1.fused', 'A.shared_offset_cand2', 'dense_i.inner.inner.inner', 'dense_shared_pos_select3', 'dense.wmma.accumulator_i.c.inner.inner.inner', 'dense_j.inner.outer', 'wmma_m_wmma_n', 'dense_i.inner.inner.inner.outer', 'dense.wmma.accumulator_i.c.inner', 'dense.wmma.accumulator_shared_pos_select0', 'dense.wmma.accumulator_local_pos_select0', 'dense.wmma.accumulator_j.c.inner.inner.inner', 'dense.wmma.accumulator.shared_offset_cand1', 'dense.wmma.accumulator.shared_align_size', 'dense_vectorize', 'B.shared_offset_cand5', 'dense_i.inner.inner.inner.inner', 'A.shared_vectorize_cand2', 'dense.wmma.accumulator_shared_pos_select1', 'dense.wmma.accumulator.shared_ax1.outer', 'A.shared_offset_cand3', 'dense_j.inner.inner.inner.outer', 'wmma_n_cand1', 'wmma_m_cand1', 'dense.wmma.accumulator_local_pos_select3', 'dense_i.inner.inner.outer', 'B.shared_offset_cand4', 'dense.wmma.accumulator.shared_offset_cand4', 'A.shared_vectorize_cand3', 'dense.wmma.accumulator.shared_ax0.inner', 'B.shared_vectorize_cand0', 'wmma_m_cand0', 'dense_j.inner.inner.inner.inner.inner', 'blockIdx.x', 'B.shared_offset_cand2', 'threads', 'B.shared_align_size', 'dense.wmma.accumulator_j.c.inner', 'dense_vectorize_cand2', 'wmma_n_cand2', 'dense_shared_pos_select4', 'A.shared_vectorize_cand1', 'dense.wmma.accumulator_k.inner', 'dense.wmma.accumulator.shared_local_pos_select0', 'dense.wmma.accumulator_i.c.inner.inner', 'dense.wmma.accumulator.shared_offset_cand3', 'dense.wmma.accumulator_k.inner.inner.inner', 'dense_shared_pos_select0', 'B.shared_offset_cand3', 'A.shared_tmp_ax0', 'dense.wmma.accumulator.shared_ax0.outer', 'dense.wmma.accumulator.shared_offset_cand2', 'dense_j.inner.inner.inner', 'A.shared_offset_cand1', 'dense.wmma.accumulator_local_pos_select1', 'A.shared_offset_cand0', 'B.shared_vectorize_cand3', 'dense_i.inner.inner', 'B.shared_tmp_ax0', 'B.shared_vectorize_cand2', 'A.shared_offset_cand5', 'dense_shared_pos_select2'}



{'densei.innertileSpatial': 1, 'densej.innertileSpatial': 1, 'densei.inner.innertileSpatial': 1, 'densej.inner.innertileSpatial': 1, 'densei.inner.inner.innertileSpatial': 1, 'densej.inner.inner.innertileSpatial': 1, 'densei.inner.inner.inner.innertileSpatial': 1, 'densej.inner.inner.inner.innertileSpatial': 1, 'dense_i.outer': 1, 'dense_j.outer': 1, 'dense_i.inner.outer.j.inner.outer.fused': 1, 'dense_i.inner.inner.outer.j.inner.inner.outer.fused': 1, 'dense_i.inner.inner.inner.outer.j.inner.inner.inner.outer.fused': 1, 'dense_i.inner.inner.inner.inner.outer': 1, 'dense_j.inner.inner.inner.inner.outer': 1, 'dense_i.inner.inner.inner.inner.inner.j.inner.inner.inner.inner.inner.fused': 1, 'dense_shared_pos': 1, 'dense.wmma.accumulator.shared_ax0': 1, 'dense.wmma.accumulator.shared_ax1': 1, 'dense.wmma.accumulator.shared_offset': 1, 'dense.wmma.accumulator.sharedax0tileSpatial': 1, 'dense.wmma.accumulator.sharedax1tileSpatial': 1, 'wmma_m': 1, 'wmma_k': 1, 'wmma_n': 1, 'dense.wmma.accumulator.shared_ax0.outer.ax1.outer.fused': 1, 'dense.wmma.accumulator.shared_ax0.inner.outer': 1, 'dense.wmma.accumulator.shared_ax1.inner.outer': 1, 'dense.wmma.accumulator.shared_ax0.inner.inner': 1, 'dense.wmma.accumulator.shared_ax1.inner.inner': 1, 'dense.wmma.accumulator.shared_local_pos': 1, 'dense.wmma.accumulator_i.c': 1, 'dense.wmma.accumulator_j.c': 1, 'dense.wmma.accumulatori.ctileAll': 1, 'dense.wmma.accumulatorj.ctileAll': 1, 'dense.wmma.accumulatorktileAll': 1, 'dense.wmma.accumulatori.c.innertileAll': 1, 'dense.wmma.accumulatorj.c.innertileAll': 1, 'dense.wmma.accumulatork.innertileAll': 1, 'dense.wmma.accumulatori.c.inner.innertileAll': 1, 'dense.wmma.accumulatorj.c.inner.innertileAll': 1, 'dense.wmma.accumulatork.inner.innertileAll': 1, 'dense.wmma.accumulator_i.c.outer': 1, 'dense.wmma.accumulator_j.c.outer': 1, 'dense.wmma.accumulator_k.outer': 1, 'dense.wmma.accumulator_i.c.inner.outer': 1, 'dense.wmma.accumulator_j.c.inner.outer': 1, 'dense.wmma.accumulator_k.inner.outer': 1, 'dense.wmma.accumulator_i.c.inner.inner.outer': 1, 'dense.wmma.accumulator_j.c.inner.inner.outer': 1, 'dense.wmma.accumulator_k.inner.inner.outer': 1, 'dense.wmma.accumulator_i.c.inner.inner.inner.outer': 1, 'dense.wmma.accumulator_j.c.inner.inner.inner.outer': 1, 'dense.wmma.accumulator_k.inner.inner.inner.outer': 1, 'dense.wmma.accumulator_i.c.inner.inner.inner.inner': 1, 'dense.wmma.accumulator_j.c.inner.inner.inner.inner': 1, 'dense.wmma.accumulator_k.inner.inner.inner.inner': 1, 'dense.wmma.accumulator_local_pos': 1, 'B.shared.wmma.matrix_b_ax0': 1, 'B.shared.wmma.matrix_b_ax1': 1, 'B.shared.wmma.matrix_b_ax0.outer': 1, 'B.shared.wmma.matrix_b_ax1.outer': 1, 'B.shared.wmma.matrix_b_ax0.inner': 1, 'B.shared.wmma.matrix_b_ax1.inner': 1, 'dense.wmma.accumulator_shared_pos': 1, 'B.shared_ax0': 1, 'B.shared_ax1': 1, 'B.shared_offset': 1, 'B.shared_vectorize': 1, 'threadIdx.x': 1, 'threadIdx.y': 1, 'A.shared.wmma.matrix_a_ax0': 1, 'A.shared.wmma.matrix_a_ax1': 1, 'A.shared.wmma.matrix_a_ax0.outer': 1, 'A.shared.wmma.matrix_a_ax1.outer': 1, 'A.shared.wmma.matrix_a_ax0.inner': 1, 'A.shared.wmma.matrix_a_ax1.inner': 1, 'A.shared_ax0': 1, 'A.shared_ax1': 1, 'A.shared_offset': 1, 'A.shared_vectorize': 1, 'dense_unroll_pragma': 1}


{'dense_i.inner.inner.inner', 'dense.wmma.accumulator.shared_shared_mem_size', 'A.shared_vectorize_cand0', 'dense_i.inner.inner.inner.inner.inner', 'dense.wmma.accumulator.shared_ax1.inner', 'B.shared_offset_cand3', 'blockIdx.x', 'dense.wmma.accumulator.shared_local_pos_select1', 'dense.wmma.accumulator.shared_align_size', 'dense.wmma.accumulator.shared_ax0.inner', 'wmma_n_cand1', 'dense.wmma.accumulator_k.inner.inner', 'dense_shared_pos_select5', 'A.shared_offset_cand4', 'dense.wmma.accumulator.shared_offset_cand5', 'dense_j.inner.inner', 'dense.wmma.accumulator_j.c.inner.inner.inner', 'A.shared_offset_cand2', 'B.shared_offset_cand0', 'A.shared_vectorize_cand2', 'B.shared_vectorize_cand3', 'A.shared_vectorize_cand1', 'dense.wmma.accumulator_k.inner', 'dense.wmma.accumulator_i.c.inner', 'dense_vectorize_cand3', 'wmma_m_cand1', 'B.shared_shared_mem_size', 'dense_i.inner.inner.inner.inner', 'dense.wmma.accumulator.shared_offset_cand0', 'dense_j.inner.inner.inner.inner.inner', 'B.shared_vectorize_cand0', 'B.shared_vectorize_cand2', 'dense.wmma.accumulator_local_pos_select0', 'dense_j.inner.outer', 'dense_shared_pos_select0', 'B.shared_offset_cand5', 'dense.wmma.accumulator_i.c.inner.inner.inner', 'dense_i.inner.inner.outer', 'A.shared_offset_cand5', 'wmma_m_cand2', 'dense.wmma.accumulator_k.inner.inner.inner', 'dense.wmma.accumulator_shared_pos_select1', 'dense.wmma.accumulator.shared_offset_cand1', 'dense.wmma.accumulator_local_pos_select1', 'dense.wmma.accumulator_j.c.inner.inner', 'B.shared_ax0.ax1.fused', 'dense_shared_pos_select4', 'wmma_n_cand0', 'dense_j.inner.inner.inner', 'A.shared_offset_cand0', 'A.shared_shared_mem_size', 'dense.wmma.accumulator_j.c.inner', 'B.shared_vectorize_cand1', 'dense_i.inner', 'dense.wmma.accumulator.shared_offset_cand2', 'B.shared_offset_cand4', 'A.shared_ax0.ax1.fused', 'dense_i.inner.outer', 'dense_vectorize_cand2', 'dense.wmma.accumulator.shared_ax1.outer', 'B.shared_offset_cand1', 'wmma_n_cand2', 'A.shared_tmp_ax0', 'dense_shared_pos_select1', 'threads', 'A.shared_align_size', 'dense.wmma.accumulator.shared_local_pos_select0', 'dense_vectorize', 'dense.wmma.accumulator.shared_ax0.outer', 'dense.wmma.accumulator_local_pos_select2', 'dense_shared_pos_select3', 'dense_j.inner.inner.inner.inner', 'A.shared_offset_cand3', 'dense_shared_pos_select2', 'dense_j.inner.inner.inner.outer', 'dense_j.inner.inner.outer', 'dense_vectorize_cand1', 'dense.wmma.accumulator_local_pos_select3', 'dense.wmma.accumulator_shared_pos_select3', 'dense_i.inner.inner.inner.outer', 'dense.wmma.accumulator.shared_offset_cand3', 'wmma_m_wmma_n', 'wmma_m_cand0', 'A.shared_vectorize_cand3', 'dense_i.inner.inner', 'B.shared_align_size', 'dense.wmma.accumulator.shared_offset_cand4', 'B.shared_tmp_ax0', 'dense.wmma.accumulator_shared_pos_select0', 'A.shared_offset_cand1', 'dense_j.inner', 'dense_vectorize_cand0', 'dense.wmma.accumulator_i.c.inner.inner', 'B.shared_offset_cand2', 'dense.wmma.accumulator_shared_pos_select2'}



{'gpu': {'max_shared_memory_per_block': 49152, 'max_threads_per_block': 1024, 'max_thread_x': 1024, 'max_thread_y': 1024, 'max_thread_z': 64}}


{'densei.innertileSpatial': 1, 'densej.innertileSpatial': 1, 'densei.inner.innertileSpatial': 1, 'densej.inner.innertileSpatial': 1, 'densei.inner.inner.innertileSpatial': 1, 'densej.inner.inner.innertileSpatial': 1, 'densei.inner.inner.inner.innertileSpatial': 1, 'densej.inner.inner.inner.innertileSpatial': 1, 'dense_shared_pos': 1, 'dense.wmma.accumulator.shared_ax1': 1, 'dense.wmma.accumulator.shared_offset': 1, 'dense.wmma.accumulator.sharedax0tileSpatial': 1, 'dense.wmma.accumulator.sharedax1tileSpatial': 1, 'wmma_m': 1, 'wmma_k': 1, 'wmma_n': 1, 'dense.wmma.accumulator.shared_local_pos': 1, 'dense.wmma.accumulatori.ctileAll': 1, 'dense.wmma.accumulatorj.ctileAll': 1, 'dense.wmma.accumulatorktileAll': 1, 'dense.wmma.accumulatori.c.innertileAll': 1, 'dense.wmma.accumulatorj.c.innertileAll': 1, 'dense.wmma.accumulatork.innertileAll': 1, 'dense.wmma.accumulatori.c.inner.innertileAll': 1, 'dense.wmma.accumulatorj.c.inner.innertileAll': 1, 'dense.wmma.accumulatork.inner.innertileAll': 1, 'dense.wmma.accumulator_local_pos': 1, 'dense.wmma.accumulator_shared_pos': 1, 'B.shared_ax1': 1, 'B.shared_offset': 1, 'B.shared_vectorize': 1, 'threadIdx.x': 1, 'threadIdx.y': 1, 'A.shared_ax1': 1, 'A.shared_offset': 1, 'A.shared_vectorize': 1, 'dense_unroll_pragma': 1}



请问一下就是我在做tvm做编译优化，我要写prompt来做调度参数优化，我做一个类似于下面这样的prompt

prefix = f"The following are examples of performance of a DecisionTree measured in accuracy and the corresponding model hyperparameter configurations."
prefix += f" The tabular dataste contains 120 samples and 4 features (0 categorical, 4 numerical)"
prefix += f" The allowable ranges for the hyperparameters are:\n"
prefix += f"- max_depth: [1, 15] (int)"
prefix += f"- max_features: [0.01, 0.99] (float, precise to 2 decimals)\n"
prefix += f"- min_impurity_decrease: [0, 0] (float, precise to 0 decimals)\n"
prefix += f"- min_samples_leaf: [0.01, 0.49] (float, precise to 2 decimals)\n"
prefix += f"- min_samples_split: [0.01, 0.99] (float, precise to 2 decimals)\n"
prefix += f"- min_weight_fraction_leaf: [0.01, 0.49] (float, precise to 2 decimals)\n"

prefix += f"Recommend a configuration that achieve the target performance of 0.95833. Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. Recommend values with highest precision, as requestedd by the allowed ranges. "
prefix += f" Your response must only contain the predicted configuration, in the format ## configuration ##.\n"

上面这个prompt我修改，我运行的平台是NVIDIA GPU platforms是the NVIDIA RTX3080，我平台的硬件约束是'gpu': {'max_shared_memory_per_block': 49152, 'max_threads_per_block': 1024, 'max_thread_x': 1024, 'max_thread_y': 1024, 'max_thread_z': 64}，然后我评价指标是perf，perf是越大越好，下面是一些调度参数设置的例子
few_shot_examples = [{'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 8, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 2, densej.inner.inner.inner.innertileSpatial: 1, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 48, dense.wmma.accumulator.sharedax0tileSpatial: 1, dense.wmma.accumulator.sharedax1tileSpatial: 8, wmma_m: 32, wmma_k: 16, wmma_n: 8, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 2, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 2, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 2, dense.wmma.accumulator_local_pos: 3, dense.wmma.accumulator_shared_pos: 2, B.shared_ax1: 16, B.shared_offset: 32, B.shared_vectorize: 2, threadIdx.x: 32, threadIdx.y: 8, A.shared_ax1: 16, A.shared_offset: 8, A.shared_vectorize: 8, dense_unroll_pragma: 4 ##', 'A': '0.0'}, 
{'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 16, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 2, densej.inner.inner.inner.innertileSpatial: 2, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 32, dense.wmma.accumulator.sharedax0tileSpatial: 8, dense.wmma.accumulator.sharedax1tileSpatial: 2, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 1, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 4, dense.wmma.accumulator_local_pos: 3, dense.wmma.accumulator_shared_pos: 3, B.shared_ax1: 16, B.shared_offset: 0, B.shared_vectorize: 2, threadIdx.x: 32, threadIdx.y: 16, A.shared_ax1: 16, A.shared_offset: 48, A.shared_vectorize: 2, dense_unroll_pragma: 4 ##', 'A': '12.541613'}, 
{'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 1, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 8, densej.inner.inner.innertileSpatial: 4, densei.inner.inner.inner.innertileSpatial: 8, densej.inner.inner.inner.innertileSpatial: 16, dense_shared_pos: 2, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 24, dense.wmma.accumulator.sharedax0tileSpatial: 1, dense.wmma.accumulator.sharedax1tileSpatial: 1, wmma_m: 32, wmma_k: 16, wmma_n: 8, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 2, dense.wmma.accumulatorj.ctileAll: 8, dense.wmma.accumulatorktileAll: 4, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 1, dense.wmma.accumulator_local_pos: 0, dense.wmma.accumulator_shared_pos: 0, B.shared_ax1: 16, B.shared_offset: 24, B.shared_vectorize: 4, threadIdx.x: 32, threadIdx.y: 1, A.shared_ax1: 16, A.shared_offset: 8, A.shared_vectorize: 2, dense_unroll_pragma: 3 ##', 'A': '11.630533'}, 
{'Q': '## densei.innertileSpatial: 2, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 4, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 2, densej.inner.inner.inner.innertileSpatial: 2, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 48, dense.wmma.accumulator.sharedax0tileSpatial: 4, dense.wmma.accumulator.sharedax1tileSpatial: 1, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 2, dense.wmma.accumulatorktileAll: 1, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 2, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 2, dense.wmma.accumulator_local_pos: 2, dense.wmma.accumulator_shared_pos: 2, B.shared_ax1: 16, B.shared_offset: 8, B.shared_vectorize: 8, threadIdx.x: 32, threadIdx.y: 4, A.shared_ax1: 16, A.shared_offset: 24, A.shared_vectorize: 1, dense_unroll_pragma: 1 ##', 'A': '12.792116'}, 
{'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 8, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 8, densej.inner.inner.innertileSpatial: 4, densei.inner.inner.inner.innertileSpatial: 1, densej.inner.inner.inner.innertileSpatial: 2, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 24, dense.wmma.accumulator.sharedax0tileSpatial: 4, dense.wmma.accumulator.sharedax1tileSpatial: 2, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 1, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 1, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 4, dense.wmma.accumulator_local_pos: 2, dense.wmma.accumulator_shared_pos: 0, B.shared_ax1: 64, B.shared_offset: 0, B.shared_vectorize: 4, threadIdx.x: 32, threadIdx.y: 8, A.shared_ax1: 64, A.shared_offset: 32, A.shared_vectorize: 2, dense_unroll_pragma: 3 ##', 'A': '12.266978'},
{'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 16, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 4, densej.inner.inner.inner.innertileSpatial: 1, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 32, dense.wmma.accumulator.sharedax0tileSpatial: 4, dense.wmma.accumulator.sharedax1tileSpatial: 4, wmma_m: 16, wmma_k: 16, wmma_n: 16, dense.wmma.accumulator.shared_local_pos: 1, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 1, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 2, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 2, dense.wmma.accumulator_local_pos: 3, dense.wmma.accumulator_shared_pos: 3, B.shared_ax1: 16, B.shared_offset: 0, B.shared_vectorize: 2, threadIdx.x: 32, threadIdx.y: 16, A.shared_ax1: 16, A.shared_offset: 24, A.shared_vectorize: 1, dense_unroll_pragma: 5 ##', 'A': '12.672532'},
{'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 8, densei.inner.innertileSpatial: 1, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 8, densej.inner.inner.innertileSpatial: 4, densei.inner.inner.inner.innertileSpatial: 1, densej.inner.inner.inner.innertileSpatial: 2, dense_shared_pos: 2, dense.wmma.accumulator.shared_ax1: 8, dense.wmma.accumulator.shared_offset: 0, dense.wmma.accumulator.sharedax0tileSpatial: 1, dense.wmma.accumulator.sharedax1tileSpatial: 1, wmma_m: 32, wmma_k: 16, wmma_n: 8, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 1, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 4, dense.wmma.accumulatori.c.inner.innertileAll: 2, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 1, dense.wmma.accumulator_local_pos: 3, dense.wmma.accumulator_shared_pos: 2, B.shared_ax1: 16, B.shared_offset: 32, B.shared_vectorize: 4, threadIdx.x: 32, threadIdx.y: 1, A.shared_ax1: 16, A.shared_offset: 8, A.shared_vectorize: 8, dense_unroll_pragma: 2 ##', 'A': '0.00000'},
{'Q': '## densei.innertileSpatial: 2, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 1, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 32, densej.inner.inner.inner.innertileSpatial: 2, dense_shared_pos: 2, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 0, dense.wmma.accumulator.sharedax0tileSpatial: 1, dense.wmma.accumulator.sharedax1tileSpatial: 1, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 2, dense.wmma.accumulatorktileAll: 4, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 4, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 1, dense.wmma.accumulator_local_pos: 2, dense.wmma.accumulator_shared_pos: 2, B.shared_ax1: 16, B.shared_offset: 16, B.shared_vectorize: 2, threadIdx.x: 32, threadIdx.y: 1, A.shared_ax1: 16, A.shared_offset: 48, A.shared_vectorize: 2, dense_unroll_pragma: 0 ##', 'A': '12.110779'},
{'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 1, densej.inner.innertileSpatial: 8, densei.inner.inner.innertileSpatial: 8, densej.inner.inner.innertileSpatial: 4, densei.inner.inner.inner.innertileSpatial: 8, densej.inner.inner.inner.innertileSpatial: 1, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 8, dense.wmma.accumulator.sharedax0tileSpatial: 8, dense.wmma.accumulator.sharedax1tileSpatial: 1, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 2, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 2, dense.wmma.accumulatork.inner.innertileAll: 2, dense.wmma.accumulator_local_pos: 1, dense.wmma.accumulator_shared_pos: 0, B.shared_ax1: 32, B.shared_offset: 16, B.shared_vectorize: 2, threadIdx.x: 32, threadIdx.y: 8, A.shared_ax1: 32, A.shared_offset: 48, A.shared_vectorize: 2, dense_unroll_pragma: 5 ##', 'A': '12.681599'},
{'Q': '## densei.innertileSpatial: 1, densej.innertileSpatial: 2, densei.inner.innertileSpatial: 1, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 2, densej.inner.inner.innertileSpatial: 16, densei.inner.inner.inner.innertileSpatial: 16, densej.inner.inner.inner.innertileSpatial: 2, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 32, dense.wmma.accumulator.shared_offset: 48, dense.wmma.accumulator.sharedax0tileSpatial: 1, dense.wmma.accumulator.sharedax1tileSpatial: 1, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 4, dense.wmma.accumulatori.c.innertileAll: 8, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 1, dense.wmma.accumulator_local_pos: 1, dense.wmma.accumulator_shared_pos: 1, B.shared_ax1: 16, B.shared_offset: 48, B.shared_vectorize: 8, threadIdx.x: 32, threadIdx.y: 1, A.shared_ax1: 16, A.shared_offset: 32, A.shared_vectorize: 2, dense_unroll_pragma: 3 ##', 'A': '11.961652'}
]
这些调度参数的范围如下
prefix += f"- densei.innertileSpatial: [1, 64] (int)"
prefix += f"- densej.innertileSpatial: [1, 64] (int)\n"
prefix += f"- densei.inner.innertileSpatial: [1, 64] (int)\n"
prefix += f"- densej.inner.innertileSpatial: [1, 64] (int)\n"
prefix += f"- densei.inner.inner.innertileSpatial: [1, 64] (int)\n"
prefix += f"- densej.inner.inner.innertileSpatial: [1, 64] (int)\n"
prefix += f"- densei.inner.inner.inner.innertileSpatial: [1, 64] (int)\n"
prefix += f"- densej.inner.inner.inner.innertileSpatial: [1, 64] (int)\n"
prefix += f"- dense_shared_pos: [1, 5](int)\n"
prefix += f"- dense.wmma.accumulator.shared_ax1: [1, 64] (int)\n"
prefix += f"- dense.wmma.accumulator.shared_offset: [0, 48] (int)\n"
prefix += f"- dense.wmma.accumulator.sharedax0tileSpatial: [1, 64] (int)\n"
prefix += f"- dense.wmma.accumulator.sharedax1tileSpatial: [1, 64] (int)\n"
prefix += f"- wmma_m: [8, 32] (int)\n"
prefix += f"- wmma_k: [16, 16] (int)\n"
prefix += f"- wmma_n: [8, 32] (int)\n"
prefix += f"- dense.wmma.accumulator.shared_local_pos: [0, 1] (int)\n"
prefix += f"- dense.wmma.accumulatori.ctileAll: [1, 64] (int)\n"
prefix += f"- dense.wmma.accumulatorj.ctileAll: [1, 64] (int)\n"
prefix += f"- dense.wmma.accumulatorktileAll: [1, 64] (int)\n"
prefix += f"- dense.wmma.accumulatori.c.innertileAll: [1, 64] (int)\n"
prefix += f"- dense.wmma.accumulatorj.c.innertileAll: [1, 64] (int)\n"
prefix += f"- dense.wmma.accumulatork.innertileAll: [1, 64] (int)\n"
prefix += f"- dense.wmma.accumulatori.c.inner.innertileAll: [1, 64] (int)\n"
prefix += f"- dense.wmma.accumulatorj.c.inner.innertileAll: [1, 64] (int)\n"
prefix += f"- dense.wmma.accumulatork.inner.innertileAll: [1, 64] (int)\n"
prefix += f"- dense.wmma.accumulator_local_pos: [0, 3] (int)\n"
prefix += f"- dense.wmma.accumulator_shared_pos: [0, 3] (int)\n"
prefix += f"- B.shared_ax1: [1, 64] (int)\n"
prefix += f"- B.shared_offset: [0, 48] (int)\n"
prefix += f"- B.shared_vectorize: [1, 8] (int)\n"
prefix += f"- threadIdx.x: [32, 32] (int)\n"
prefix += f"- threadIdx.y: [1, 1024] (int)\n"
prefix += f"- A.shared_ax1: [1, 64] (int)\n"
prefix += f"- A.shared_offset: [0, 48] (int)\n"
prefix += f"- A.shared_vectorize: [1, 8] (int)\n"
prefix += f"- dense_unroll_pragma: [0, 5] (int)\n"



prefix = f"The following are examples of TVM schedule parameter configurations and their corresponding performance metrics (perf) on an NVIDIA RTX3080 GPU. Higher perf values are better."
prefix += f"\n\nHardware constraints for NVIDIA RTX3080:"
prefix += f"\n- max_shared_memory_per_block: 49152"
prefix += f"\n- max_threads_per_block: 1024"
prefix += f"\n- max_thread_x: 1024"
prefix += f"\n- max_thread_y: 1024"
prefix += f"\n- max_thread_z: 64"

prefix += f"\n\nThe allowable ranges for the schedule parameters are:"
prefix += f"\n- densei.innertileSpatial: [1, 64] (int)"
prefix += f"\n- densej.innertileSpatial: [1, 64] (int)"
prefix += f"\n- densei.inner.innertileSpatial: [1, 64] (int)"
prefix += f"\n- densej.inner.innertileSpatial: [1, 64] (int)"
prefix += f"\n- densei.inner.inner.innertileSpatial: [1, 64] (int)"
prefix += f"\n- densej.inner.inner.innertileSpatial: [1, 64] (int)"
prefix += f"\n- densei.inner.inner.inner.innertileSpatial: [1, 64] (int)"
prefix += f"\n- densej.inner.inner.inner.innertileSpatial: [1, 64] (int)"
prefix += f"\n- dense_shared_pos: [1, 5] (int)"
prefix += f"\n- dense.wmma.accumulator.shared_ax1: [1, 64] (int)"
prefix += f"\n- dense.wmma.accumulator.shared_offset: [0, 48] (int)"
prefix += f"\n- dense.wmma.accumulator.sharedax0tileSpatial: [1, 64] (int)"
prefix += f"\n- dense.wmma.accumulator.sharedax1tileSpatial: [1, 64] (int)"
prefix += f"\n- wmma_m: [8, 32] (int)"
prefix += f"\n- wmma_k: [16, 16] (int)" 
prefix += f"\n- wmma_n: [8, 32] (int)"
prefix += f"\n- dense.wmma.accumulator.shared_local_pos: [0, 1] (int)"
prefix += f"\n- dense.wmma.accumulatori.ctileAll: [1, 64] (int)"
prefix += f"\n- dense.wmma.accumulatorj.ctileAll: [1, 64] (int)"
prefix += f"\n- dense.wmma.accumulatorktileAll: [1, 64] (int)"
prefix += f"\n- dense.wmma.accumulatori.c.innertileAll: [1, 64] (int)"
prefix += f"\n- dense.wmma.accumulatorj.c.innertileAll: [1, 64] (int)"
prefix += f"\n- dense.wmma.accumulatork.innertileAll: [1, 64] (int)"
prefix += f"\n- dense.wmma.accumulatori.c.inner.innertileAll: [1, 64] (int)"
prefix += f"\n- dense.wmma.accumulatorj.c.inner.innertileAll: [1, 64] (int)"
prefix += f"\n- dense.wmma.accumulatork.inner.innertileAll: [1, 64] (int)"
prefix += f"\n- dense.wmma.accumulator_local_pos: [0, 3] (int)"
prefix += f"\n- dense.wmma.accumulator_shared_pos: [0, 3] (int)"
prefix += f"\n- B.shared_ax1: [1, 64] (int)"
prefix += f"\n- B.shared_offset: [0, 48] (int)"
prefix += f"\n- B.shared_vectorize: [1, 8] (int)"
prefix += f"\n- threadIdx.x: [32, 32] (int)"
prefix += f"\n- threadIdx.y: [1, 1024] (int)"
prefix += f"\n- A.shared_ax1: [1, 64] (int)"
prefix += f"\n- A.shared_offset: [0, 48] (int)"
prefix += f"\n- A.shared_vectorize: [1, 8] (int)"
prefix += f"\n- dense_unroll_pragma: [0, 5] (int)"

prefix += f"\n\nBelow are examples of configurations and their corresponding performance (perf) values:"
prefix += f"\n\nConfiguration 1 (perf: 0.0):"
prefix += f"\ndensei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 8, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 2, densej.inner.inner.inner.innertileSpatial: 1, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 48, dense.wmma.accumulator.sharedax0tileSpatial: 1, dense.wmma.accumulator.sharedax1tileSpatial: 8, wmma_m: 32, wmma_k: 16, wmma_n: 8, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 2, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 2, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 2, dense.wmma.accumulator_local_pos: 3, dense.wmma.accumulator_shared_pos: 2, B.shared_ax1: 16, B.shared_offset: 32, B.shared_vectorize: 2, threadIdx.x: 32, threadIdx.y: 8, A.shared_ax1: 16, A.shared_offset: 8, A.shared_vectorize: 8, dense_unroll_pragma: 4"

prefix += f"\n\nConfiguration 2 (perf: 12.541613):"
prefix += f"\ndensei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 16, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 2, densej.inner.inner.inner.innertileSpatial: 2, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 32, dense.wmma.accumulator.sharedax0tileSpatial: 8, dense.wmma.accumulator.sharedax1tileSpatial: 2, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 1, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 4, dense.wmma.accumulator_local_pos: 3, dense.wmma.accumulator_shared_pos: 3, B.shared_ax1: 16, B.shared_offset: 0, B.shared_vectorize: 2, threadIdx.x: 32, threadIdx.y: 16, A.shared_ax1: 16, A.shared_offset: 48, A.shared_vectorize: 2, dense_unroll_pragma: 4"

prefix += f"\n\nConfiguration 3 (perf: 12.792116):"
prefix += f"\ndensei.innertileSpatial: 2, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 4, densej.inner.innertileSpatial: 1, densei.inner.inner.innertileSpatial: 1, densej.inner.inner.innertileSpatial: 32, densei.inner.inner.inner.innertileSpatial: 2, densej.inner.inner.inner.innertileSpatial: 2, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 48, dense.wmma.accumulator.sharedax0tileSpatial: 4, dense.wmma.accumulator.sharedax1tileSpatial: 1, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 2, dense.wmma.accumulatorktileAll: 1, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 2, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 1, dense.wmma.accumulatork.inner.innertileAll: 2, dense.wmma.accumulator_local_pos: 2, dense.wmma.accumulator_shared_pos: 2, B.shared_ax1: 16, B.shared_offset: 8, B.shared_vectorize: 8, threadIdx.x: 32, threadIdx.y: 4, A.shared_ax1: 16, A.shared_offset: 24, A.shared_vectorize: 1, dense_unroll_pragma: 1"

prefix += f"\n\nConfiguration 4 (perf: 12.681599):"
prefix += f"\ndensei.innertileSpatial: 1, densej.innertileSpatial: 1, densei.inner.innertileSpatial: 1, densej.inner.innertileSpatial: 8, densei.inner.inner.innertileSpatial: 8, densej.inner.inner.innertileSpatial: 4, densei.inner.inner.inner.innertileSpatial: 8, densej.inner.inner.inner.innertileSpatial: 1, dense_shared_pos: 1, dense.wmma.accumulator.shared_ax1: 64, dense.wmma.accumulator.shared_offset: 8, dense.wmma.accumulator.sharedax0tileSpatial: 8, dense.wmma.accumulator.sharedax1tileSpatial: 1, wmma_m: 8, wmma_k: 16, wmma_n: 32, dense.wmma.accumulator.shared_local_pos: 0, dense.wmma.accumulatori.ctileAll: 1, dense.wmma.accumulatorj.ctileAll: 1, dense.wmma.accumulatorktileAll: 2, dense.wmma.accumulatori.c.innertileAll: 1, dense.wmma.accumulatorj.c.innertileAll: 1, dense.wmma.accumulatork.innertileAll: 1, dense.wmma.accumulatori.c.inner.innertileAll: 1, dense.wmma.accumulatorj.c.inner.innertileAll: 2, dense.wmma.accumulatork.inner.innertileAll: 2, dense.wmma.accumulator_local_pos: 1, dense.wmma.accumulator_shared_pos: 0, B.shared_ax1: 32, B.shared_offset: 16, B.shared_vectorize: 2, threadIdx.x: 32, threadIdx.y: 8, A.shared_ax1: 32, A.shared_offset: 48, A.shared_vectorize: 2, dense_unroll_pragma: 5"

prefix += f"\n\nBased on the examples above, recommend a new TVM schedule parameter configuration that will achieve a high performance value (perf > 12.8). The configuration should respect all hardware constraints and parameter ranges specified above. Do not recommend values at minimum or maximum of allowable ranges unless necessary, and ensure the configuration is optimal for the NVIDIA RTX3080 GPU architecture."

prefix += f"\n\nYour response must only contain the predicted configuration, in the format ## configuration ##."


假设先生成一个，先不考虑生成问题



 # from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((64, 64), "float16"), B: T.Buffer((64, 64), "float16"), dense: T.Buffer((64, 64), "float16")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
        dense_wmma_accumulator_shared = T.allocate([4096], "float16", "shared")
        with T.launch_thread("threadIdx.y", 1) as threadIdx_y:
            dense_wmma_accumulator = T.allocate([4096], "float16", "wmma.accumulator")
            A_shared = T.allocate([4096], "float16", "shared")
            A_shared_wmma_matrix_a = T.allocate([4096], "float16", "wmma.matrix_a")
            B_shared_wmma_matrix_b = T.allocate([4096], "float16", "wmma.matrix_b")
            for i_c_inner_inner_inner_outer_init, j_c_inner_inner_inner_outer_init in T.grid(4, 4):
                T.tvm_fill_fragment(dense_wmma_accumulator, 16, 16, 16, i_c_inner_inner_inner_outer_init * 4 + j_c_inner_inner_inner_outer_init, T.float32(0))
            for ax0_ax1_fused_outer_outer_outer in range(4096):
                threadIdx_y_1 = T.launch_thread("threadIdx.y", 1)
                threadIdx_x = T.launch_thread("threadIdx.x", 1)
                A_shared_1 = T.Buffer((4096,), "float16", data=A_shared, scope="shared")
                A_1 = T.Buffer((4096,), "float16", data=A.data)
                A_shared_1[ax0_ax1_fused_outer_outer_outer] = A_1[ax0_ax1_fused_outer_outer_outer]
            for ax0_outer, ax1_outer in T.grid(4, 4):
                T.tvm_load_matrix_sync(A_shared_wmma_matrix_a, 16, 16, 16, ax0_outer * 4 + ax1_outer, T.tvm_access_ptr(T.type_annotation("float16"), A_shared, ax0_outer * 1024 + ax1_outer * 16, 1024, 1), 64, "row_major")
            for ax0_ax1_fused_outer_outer_outer in range(4096):
                threadIdx_y_1 = T.launch_thread("threadIdx.y", 1)
                threadIdx_x = T.launch_thread("threadIdx.x", 1)
                A_shared_1 = T.Buffer((4096,), "float16", data=A_shared, scope="shared")
                B_1 = T.Buffer((4096,), "float16", data=B.data)
                A_shared_1[ax0_ax1_fused_outer_outer_outer] = B_1[ax0_ax1_fused_outer_outer_outer]
            for ax0_outer, ax1_outer in T.grid(4, 4):
                T.tvm_load_matrix_sync(B_shared_wmma_matrix_b, 16, 16, 16, ax0_outer * 4 + ax1_outer, T.tvm_access_ptr(T.type_annotation("float16"), A_shared, ax0_outer * 1024 + ax1_outer * 16, 1024, 1), 64, "col_major")
            for i_c_inner_inner_inner_outer, j_c_inner_inner_inner_outer, k_inner_inner_inner_outer in T.grid(4, 4, 4):
                cse_var_2: T.int32 = i_c_inner_inner_inner_outer * 4
                cse_var_1: T.int32 = cse_var_2 + j_c_inner_inner_inner_outer
                T.tvm_mma_sync(dense_wmma_accumulator, cse_var_1, A_shared_wmma_matrix_a, cse_var_2 + k_inner_inner_inner_outer, B_shared_wmma_matrix_b, j_c_inner_inner_inner_outer * 4 + k_inner_inner_inner_outer, dense_wmma_accumulator, cse_var_1)
            for ax0_inner_outer, ax1_inner_outer in T.grid(4, 4):
                T.tvm_store_matrix_sync(dense_wmma_accumulator, 16, 16, 16, ax0_inner_outer * 4 + ax1_inner_outer, T.tvm_access_ptr(T.type_annotation("float16"), dense_wmma_accumulator_shared, ax0_inner_outer * 1024 + ax1_inner_outer * 16, 1024, 2), 64, "row_major")
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_y = T.launch_thread("threadIdx.y", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 1)
        dense_1 = T.Buffer((4096,), "float16", data=dense.data)
        dense_wmma_accumulator_shared_1 = T.Buffer((4096,), "float16", data=dense_wmma_accumulator_shared, scope="shared")
        dense_1[0:4096] = dense_wmma_accumulator_shared_1[0:4096]


# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((64, 64), "float16"), B: T.Buffer((64, 64), "float16"), dense: T.Buffer((64, 64), "float16")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
        blockIdx_x = T.launch_thread("blockIdx.x", 8)
        dense_wmma_accumulator = T.allocate([256], "float16", "wmma.accumulator")
        A_shared = T.allocate([2560], "float16", "shared")
        A_shared_wmma_matrix_a = T.allocate([512], "float16", "wmma.matrix_a")
        B_shared_wmma_matrix_b = T.allocate([128], "float16", "wmma.matrix_b")
        with T.launch_thread("threadIdx.y", 2) as threadIdx_y:
            T.tvm_fill_fragment(dense_wmma_accumulator, 32, 8, 16, 0, T.float32(0))
            for k_outer in range(4):
                for ax0_ax1_fused_outer_outer_outer in T.unroll(4):
                    threadIdx_y_1 = T.launch_thread("threadIdx.y", 2)
                    threadIdx_x = T.launch_thread("threadIdx.x", 32)
                    A_shared_1 = T.Buffer((2560,), "float16", data=A_shared, scope="shared")
                    A_1 = T.Buffer((4096,), "float16", data=A.data)
                    A_shared_1[ax0_ax1_fused_outer_outer_outer * 640 + threadIdx_y_1 * 320 + threadIdx_x // 4 * 40 + threadIdx_x % 4 * 4:ax0_ax1_fused_outer_outer_outer * 640 + threadIdx_y_1 * 320 + threadIdx_x // 4 * 40 + threadIdx_x % 4 * 4 + 4] = A_1[ax0_ax1_fused_outer_outer_outer * 1024 + threadIdx_y_1 * 512 + threadIdx_x // 4 * 64 + k_outer * 16 + threadIdx_x % 4 * 4:ax0_ax1_fused_outer_outer_outer * 1024 + threadIdx_y_1 * 512 + threadIdx_x // 4 * 64 + k_outer * 16 + threadIdx_x % 4 * 4 + 4]
                T.tvm_load_matrix_sync(A_shared_wmma_matrix_a, 32, 8, 16, 0, T.tvm_access_ptr(T.type_annotation("float16"), A_shared, threadIdx_y * 1280, 1280, 1), 40, "row_major")
                with T.launch_thread("threadIdx.y", 2) as threadIdx_y_1:
                    threadIdx_x = T.launch_thread("threadIdx.x", 32)
                    A_shared_1 = T.Buffer((256,), "float16", data=A_shared, scope="shared")
                    B_1 = T.Buffer((4096,), "float16", data=B.data)
                    A_shared_1[threadIdx_y_1 * 128 + threadIdx_x // 8 * 32 + threadIdx_x % 8 * 2:threadIdx_y_1 * 128 + threadIdx_x // 8 * 32 + threadIdx_x % 8 * 2 + 2] = B_1[blockIdx_x * 512 + threadIdx_y_1 * 256 + threadIdx_x // 8 * 64 + k_outer * 16 + threadIdx_x % 8 * 2:blockIdx_x * 512 + threadIdx_y_1 * 256 + threadIdx_x // 8 * 64 + k_outer * 16 + threadIdx_x % 8 * 2 + 2]
                T.tvm_load_matrix_sync(B_shared_wmma_matrix_b, 32, 8, 16, 0, T.tvm_access_ptr(T.type_annotation("float16"), A_shared, 0, 256, 1), 32, "col_major")
                T.tvm_mma_sync(dense_wmma_accumulator, 0, A_shared_wmma_matrix_a, 0, B_shared_wmma_matrix_b, 0, dense_wmma_accumulator, 0)
            T.tvm_store_matrix_sync(dense_wmma_accumulator, 32, 8, 16, 0, T.tvm_access_ptr(T.type_annotation("float16"), A_shared, threadIdx_y * 1024, 1024, 2), 32, "row_major")
        threadIdx_y = T.launch_thread("threadIdx.y", 2)
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        dense_1 = T.Buffer((4096,), "float16", data=dense.data)
        A_shared_1 = T.Buffer((2048,), "float16", data=A_shared, scope="shared")
        dense_1[threadIdx_y * 2048 + threadIdx_x * 64 + blockIdx_x * 8:threadIdx_y * 2048 + threadIdx_x * 64 + blockIdx_x * 8 + 8] = A_shared_1[threadIdx_y * 1024 + threadIdx_x * 32:threadIdx_y * 1024 + threadIdx_x * 32 + 8]



{'wmma_m': {'min': 8, 'max': 32}, 'wmma_k': {'min': 16, 'max': 16}, 'wmma_n': {'min': 8, 'max': 32}, 'dense.wmma.accumulator_shared_pos': {'min': 0, 'max': 10}, 'dense.wmma.accumulator_local_pos': {'min': 0, 'max': 10}, 'dense_shared_pos': {'min': 0, 'max': 10}, 'dense.wmma.accumulator.shared_local_pos': {'min': 0, 'max': 10}, 'dense_unroll_pragma': {'min': 0, 'max': 5}, 'densei.innertileSpatial': {'min': 1, 'max': 64}, 'densej.innertileSpatial': {'min': 1, 'max': 64}, 'densei.inner.innertileSpatial': {'min': 1, 'max': 64}, 'densej.inner.innertileSpatial': {'min': 1, 'max': 64}, 'threadIdx.y': {'min': 1, 'max': 1024}, 'densei.inner.inner.innertileSpatial': {'min': 1, 'max': 64}, 'densej.inner.inner.innertileSpatial': {'min': 1, 'max': 64}, 'threadIdx.x': {'min': 1, 'max': 1024}, 'densei.inner.inner.inner.innertileSpatial': {'min': 1, 'max': 64}, 'densej.inner.inner.inner.innertileSpatial': {'min': 1, 'max': 64}, 'dense.wmma.accumulator.shared_offset': {'min': 0, 'max': 48}, 'dense.wmma.accumulator.sharedax0tileSpatial': {'min': 1, 'max': 64}, 'dense.wmma.accumulator.sharedax1tileSpatial': {'min': 1, 'max': 64}, 'dense.wmma.accumulatori.ctileAll': {'min': 1, 'max': 64}, 'dense.wmma.accumulatorj.ctileAll': {'min': 1, 'max': 64}, 'dense.wmma.accumulatorktileAll': {'min': 1, 'max': 64}, 'dense.wmma.accumulatori.c.innertileAll': {'min': 1, 'max': 64}, 'dense.wmma.accumulatorj.c.innertileAll': {'min': 1, 'max': 64}, 'dense.wmma.accumulatork.innertileAll': {'min': 1, 'max': 64}, 'dense.wmma.accumulatori.c.inner.innertileAll': {'min': 1, 'max': 64}, 'dense.wmma.accumulatorj.c.inner.innertileAll': {'min': 1, 'max': 64}, 'dense.wmma.accumulatork.inner.innertileAll': {'min': 1, 'max': 64}, 'B.shared_offset': {'min': 0, 'max': 48}, 'B.shared_vectorize': {'min': 1, 'max': 8}, 'A.shared_offset': {'min': 0, 'max': 48}, 'A.shared_vectorize': {'min': 1, 'max': 8}}


{'densei.innertileSpatial': 1, 'densej.innertileSpatial': 1, 'densei.inner.innertileSpatial': 1, 'densej.inner.innertileSpatial': 1, 'densei.inner.inner.innertileSpatial': 1, 'densej.inner.inner.innertileSpatial': 1, 'densei.inner.inner.inner.innertileSpatial': 1, 'densej.inner.inner.inner.innertileSpatial': 1, 'dense_shared_pos': 1, 'dense.wmma.accumulator.shared_ax1': 1, 'dense.wmma.accumulator.shared_offset': 1, 'dense.wmma.accumulator.sharedax0tileSpatial': 1, 'dense.wmma.accumulator.sharedax1tileSpatial': 1, 'wmma_m': 1, 'wmma_k': 1, 'wmma_n': 1, 'dense.wmma.accumulator.shared_local_pos': 1, 'dense.wmma.accumulatori.ctileAll': 1, 'dense.wmma.accumulatorj.ctileAll': 1, 'dense.wmma.accumulatorktileAll': 1, 'dense.wmma.accumulatori.c.innertileAll': 1, 'dense.wmma.accumulatorj.c.innertileAll': 1, 'dense.wmma.accumulatork.innertileAll': 1, 'dense.wmma.accumulatori.c.inner.innertileAll': 1, 'dense.wmma.accumulatorj.c.inner.innertileAll': 1, 'dense.wmma.accumulatork.inner.innertileAll': 1, 'dense.wmma.accumulator_local_pos': 1, 'dense.wmma.accumulator_shared_pos': 1, 'B.shared_ax1': 1, 'B.shared_offset': 1, 'B.shared_vectorize': 1, 'threadIdx.x': 1, 'threadIdx.y': 1, 'A.shared_ax1': 1, 'A.shared_offset': 1, 'A.shared_vectorize': 1, 'dense_unroll_pragma': 1}




参数量设置
C2D
[16, 56, 56, 64, 64, 3, 3, 1, 1, 1, 'float16', 'float16']   38


bmm 
[12, 512, 64, 512, 'float16', 'float16'] 45


C1D
[16, 892, 512, 512, 3, 1, 4, 1, 'float16', 'float16']   38

C3D
                38

dil
[16, 56, 56, 64, 64, 3, 3, 1, 1, 2, 'float16', 'float16'] 59

gemm
[1024, 1024, 1024, 'float16', 'float16'] 37


scan
[16, 512, 128, 'float16', 'float16'] 37


t2d
[16, 56, 56, 64, 64, 3, 3, 1, 1, 1, 0, 'float16', 'float16']  38







task.py     apply_best

print(f"self.knob_manager.knob_names: {self.knob_manager.knob_names}")
print(f"self.knob_manager.solver.vals: {self.knob_manager.solver.vals}")
print(f"self.knob_manager.candidates: {self.knob_manager.candidates}")
print(f"self.knob_manager.sched_cand: {self.knob_manager.sched_cand}")
print(f"number solver.vals: {len(self.knob_manager.solver.vals)}")
print(f"number candidates: {len(self.knob_manager.candidates)}")
print(f"number sched_cand: {len(self.knob_manager.sched_cand)}")
print(f"self.knob_manager.sched_tups: {self.knob_manager.sched_tups}")
self.knob_manager.sched_val_state = True


































