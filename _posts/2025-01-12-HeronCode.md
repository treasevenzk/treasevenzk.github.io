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




(tvm-build-venv) newuser@cx-software:~/Heron/Heron/tests/quick_start/dlboost$ python run.py -p dlboost -c quick_start.json
Heron device  0
[64, 64, 64]
{'checks': {}}
OPS/GEMM/G1 not exists, create one
[64, 64, 64, 'int8', 'int32']
= ---- Tuning : Trials 32, Rounds 2 ---- =
== Current Round  0
PMAX 1.000000, PMin 1.000000, NUM 200
Best predicted  1.0
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
INFO:numexpr.utils:Note: NumExpr detected 20 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO:numexpr.utils:NumExpr defaulting to 16 threads.
[08:38:21] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
[08:38:22] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
[08:38:23] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
[08:38:23] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
[08:38:24] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
[08:38:24] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
[08:38:25] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
[08:38:25] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
[08:38:30] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
[08:38:30] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
[08:38:31] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
[08:38:31] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
TASK G1, Cur best 11.481211
Sample 1_3_64_1_64_64_1_64_1_1_64_1_64_1_1_64_1_64_1_64_0_1_0_0_64_4_4_16_4_16_64_4_16_4_4_4_4_4_4_4_1_4_1_4_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 0_5_16_1_16_16_1_16_4_4_4_4_4_16_4_1_4_1_1_16_1_0_0_0_16_1_1_16_1_16_1_1_1_1_16_1_16_1_1_1_16_1_16_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 0_4_2_1_2_2_1_2_1_1_2_1_2_1_1_2_1_2_0_2_1_0_0_0_2_1_1_2_1_2_1_1_1_1_2_1_2_1_1_1_2_1_2_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 2_4_64_1_64_64_1_64_1_1_64_1_64_1_1_64_1_64_1_64_0_0_1_0_64_4_4_16_4_16_64_4_16_4_4_4_4_4_4_4_1_4_1_4_1_
Perf 9.805451 , Predict 1.000000, Prob 0.000000
Sample 2_4_64_1_64_64_1_64_1_1_64_1_64_1_1_64_1_64_0_64_0_0_1_0_64_4_4_16_4_16_64_4_16_4_4_4_4_4_4_4_1_4_1_4_1_
Perf 9.815583 , Predict 1.000000, Prob 0.000000
Sample 1_2_4_1_4_4_1_4_2_2_2_2_2_4_2_1_2_1_0_2_0_1_0_0_2_1_1_2_1_2_1_1_1_1_2_1_2_1_1_1_2_1_2_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 0_1_2_1_2_2_1_2_1_1_2_1_2_1_1_2_1_2_0_2_1_0_0_0_2_1_1_2_1_2_1_1_1_1_2_1_2_1_1_1_2_1_2_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 1_4_2_1_2_2_1_2_1_1_2_1_2_1_1_2_1_2_1_2_0_1_0_0_2_1_1_2_1_2_1_1_1_1_2_1_2_1_1_1_2_1_2_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 3_3_4_1_4_4_1_4_2_2_2_2_2_4_2_1_2_1_0_1_0_0_0_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 0_2_4_1_4_4_1_4_2_2_2_2_2_4_2_1_2_1_1_4_1_0_0_0_4_1_1_4_1_4_1_1_1_1_4_1_4_1_1_1_4_1_4_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 3_2_4_1_4_4_1_4_2_2_2_2_2_4_2_1_2_1_1_1_0_0_0_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 3_2_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_0_1_0_0_0_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 3_3_64_1_64_64_1_64_8_8_8_8_8_64_8_1_8_1_1_1_0_0_0_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 0_4_2_1_2_2_1_2_1_1_2_1_2_1_1_2_1_2_1_2_1_0_0_0_2_1_1_2_1_2_1_1_1_1_2_1_2_1_1_1_2_1_2_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 3_2_64_1_64_64_1_64_8_8_8_8_8_64_8_1_8_1_0_1_0_0_0_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 0_3_4_1_4_4_1_4_2_2_2_2_2_4_2_1_2_1_1_4_1_0_0_0_4_1_1_4_1_4_1_1_1_1_4_1_4_1_1_1_4_1_4_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 2_0_64_1_64_64_1_64_8_8_8_8_8_64_8_1_8_1_1_1_0_0_1_0_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 1_5_4_1_4_4_1_4_2_2_2_2_2_4_2_1_2_1_0_2_0_1_0_0_2_1_1_2_1_2_1_1_1_1_2_1_2_1_1_1_2_1_2_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 1_2_32_1_32_32_1_32_4_4_8_4_8_16_4_2_4_2_1_8_0_1_0_0_8_2_2_4_2_4_8_2_4_2_2_2_2_2_2_2_1_2_1_2_1_
Perf 11.481211 , Predict 1.000000, Prob 0.000000
Sample 0_4_16_1_16_16_1_16_4_4_4_4_4_16_4_1_4_1_0_16_1_0_0_0_16_1_1_16_1_16_1_1_1_1_16_1_16_1_1_1_16_1_16_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_0_1_0_1_0_0_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 0_3_4_1_4_4_1_4_2_2_2_2_2_4_2_1_2_1_0_4_1_0_0_0_4_1_1_4_1_4_1_1_1_1_4_1_4_1_1_1_4_1_4_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 2_2_16_1_16_16_1_16_4_4_4_4_4_16_4_1_4_1_0_1_0_0_1_0_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 2_5_2_1_2_2_1_2_1_1_2_1_2_1_1_2_1_2_0_2_0_0_1_0_2_1_1_2_1_2_1_1_1_1_2_1_2_1_1_1_2_1_2_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 0_1_4_1_4_4_1_4_2_2_2_2_2_4_2_1_2_1_1_4_1_0_0_0_4_1_1_4_1_4_1_1_1_1_4_1_4_1_1_1_4_1_4_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 1_5_64_1_64_64_1_64_8_8_8_8_8_64_8_1_8_1_1_8_0_1_0_0_8_2_2_4_2_4_8_2_4_2_2_2_2_2_2_2_1_2_1_2_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 1_0_64_1_64_64_1_64_1_1_64_1_64_1_1_64_1_64_0_64_0_1_0_0_64_4_4_16_4_16_64_4_16_4_4_4_4_4_4_4_1_4_1_4_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 1_0_1_1_1_2_1_2_1_1_1_1_2_1_1_1_1_2_0_1_0_1_0_0_2_1_1_1_1_2_1_1_1_1_1_1_2_1_1_1_1_1_2_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 2_2_8_1_8_4_1_4_2_2_4_2_2_4_2_2_2_1_0_2_0_0_1_0_1_1_1_2_1_1_1_1_1_1_2_1_1_1_1_1_2_1_1_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 1_4_16_1_16_16_1_16_4_4_4_4_4_16_4_1_4_1_0_4_0_1_0_0_4_1_1_4_1_4_1_1_1_1_4_1_4_1_1_1_4_1_4_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 3_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_0_0_0_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Sample 2_2_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_0_0_1_0_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_1_
Perf 0.000000 , Predict 1.000000, Prob 0.000000
Hardware time 21.190347
XGB iter:   0   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:   1   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:   2   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:   3   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:   4   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:   5   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:   6   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:   7   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:   8   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:   9   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:  10   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:  11   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:  12   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:  13   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:  14   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:  15   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:  16   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:  17   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:  18   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:  19   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
XGB iter:  20   a-recall@30: 0.038889
Warning: Metric 'tr-a-recall@30' not found, using 'a-recall@30' instead.
== Current Round  1
PMAX 0.995979, PMin 0.995979, NUM 200
Time 22.318733, MAX 0.995979, TOP5_MEAN 0.995979
Time 22.574145, MAX 0.995979, TOP5_MEAN 0.995979
Time 22.970009, MAX 0.995979, TOP5_MEAN 0.995979
Time 23.378750, MAX 0.995979, TOP5_MEAN 0.995979
Time 23.662355, MAX 0.995979, TOP5_MEAN 0.995979
Best predicted  0.9959791
Heron tune time spent  24.689673900604248
[08:38:35] /home/newuser/tvm/src/runtime/threading_backend.cc:343: Warning: more than two frequencies detected! Forced big_count_ to 16
PASS
Case [64, 64, 64], latency 0.010276 ms.


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


prepare work
ctx.default_sharedload_stages: ['A.shared', 'B.shared']
ctx.shared_load_stages: ['A.shared', 'B.shared', 'dense.wmma.accumulator.shared']
ctx.no_schedule_stages: []
s.outputs: [compute(dense, body=[T.reduce(T.comm_reducer(lambda x, y: x + y, [T.float16(0)]), source=[A[i, k] * B[j, k]], init=[], axis=[T.iter_var(k, T.Range(0, 64), "CommReduce", "")], condition=T.bool(True), value_index=0)], axis=[T.iter_var(i, T.Range(0, 64), "DataPar", ""), T.iter_var(j, T.Range(0, 64), "DataPar", "")], reduce_axis=[T.iter_var(k, T.Range(0, 64), "CommReduce", "")], tag=batch_matmul, attrs={})]
stage: stage(A, placeholder(A, 0x39735010))
stage: stage(A.shared, compute(A.shared, body=[A[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 64), "DataPar", ""), T.iter_var(ax1, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=, attrs={}))
stage: stage(A.shared.wmma.matrix_a, compute(A.shared.wmma.matrix_a, body=[A.shared[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 64), "DataPar", ""), T.iter_var(ax1, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=, attrs={}))
stage: stage(B, placeholder(B, 0x39400220))
stage: stage(B.shared, compute(B.shared, body=[B[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 64), "DataPar", ""), T.iter_var(ax1, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=, attrs={}))
stage: stage(B.shared.wmma.matrix_b, compute(B.shared.wmma.matrix_b, body=[B.shared[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 64), "DataPar", ""), T.iter_var(ax1, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=, attrs={}))
stage: stage(dense.wmma.accumulator, compute(dense.wmma.accumulator, body=[T.reduce(T.comm_reducer(lambda x, y: x + y, [T.float16(0)]), source=[A.shared.wmma.matrix_a[i_c, k] * B.shared.wmma.matrix_b[j_c, k]], init=[], axis=[T.iter_var(k, T.Range(0, 64), "CommReduce", "")], condition=T.bool(True), value_index=0)], axis=[T.iter_var(i_c, T.Range(0, 64), "DataPar", ""), T.iter_var(j_c, T.Range(0, 64), "DataPar", "")], reduce_axis=[T.iter_var(k, T.Range(0, 64), "CommReduce", "")], tag=batch_matmul, attrs={}))
stage: stage(dense.wmma.accumulator.shared, compute(dense.wmma.accumulator.shared, body=[dense.wmma.accumulator[ax0, ax1]], axis=[T.iter_var(ax0, T.Range(0, 64), "DataPar", ""), T.iter_var(ax1, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=, attrs={}))
stage: stage(dense, compute(dense, body=[dense.wmma.accumulator.shared[i, j]], axis=[T.iter_var(i, T.Range(0, 64), "DataPar", ""), T.iter_var(j, T.Range(0, 64), "DataPar", "")], reduce_axis=[], tag=batch_matmul, attrs={}))



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
ctx.vectorize_stages: dense


split里面关于dense_i是否要的问题，get_ax(ax_key) 在unrollPragma作者提供的样例是64



规则生成
my test:                                        样例
EQ * 3                                          EQ * 3
sum                                             sum
EQ * 3                                          EQ * 3
sum                                             sum
ProdTwo * 2                                     ProdTwo * 2
addCacheTensorCoreOp EQ、SUM、ProdTwo

ProdTwo、EQ * 2                                 ProdTwo、EQ * 2
unrollPragma: ProdTwo、EQ


ProdTwo                                         ProdTwo
EQ、ProdTwo * 2                                 EQ、ProdTwo * 2
ProdTwo * 2                                     ProdTwo * 2
EQ、ProdTwo * 2                                 EQ、ProdTwo * 2 
ProdTow * 2                                     ProdTow * 2
EQ、ProdTwo * 2                                 EQ、ProdTwo * 2 
ProdTwo                                         ProdTwo
EQ*4                                            EQ*4
sum                                             sum
ProdTwo、EQ * 3                                 ProdTwo、EQ * 3
NE                                              NE
EQ * 12                                         EQ * 4
sum                                             sum
EQ * 12                                         EQ * 4
sum                                             sum
EQ * 6                                          EQ * 6
sum * 2                                         sum * 2
ProdTwo、EQ * 2                                 ProdTwo、EQ * 2
ProdTwo * 2                                     ProdTwo * 2 
ProdTwo、EQ * 2                                 ProdTwo、EQ * 2
EQ * 4                                          EQ * 2
sum                                             sum
EQ * 4                                          EQ * 2
sum                                             sum 
ProdTwo、EQ * 12                                ProdTwo、EQ * 12 
LE                                              LE
ProdTwo、EQ * 2                                 ProdTwo、EQ * 2
EQ * 8                                          EQ * 8
sum                                             sum
EQ * 8                                          EQ * 8 
sum                                             sum
EQ * 6                                          EQ * 6
sum * 2                                         sum * 2
ProdTwo                                         ProdTwo
EQ * 4                                          EQ * 4
sum                                             sum
EQ * 8                                          EQ * 8
sum                                             sum
EQ * 8                                          EQ * 8
sum                                             sum 
ProdTwo、EQ * 2                                 ProdTwo、EQ * 2
EQ * 8                                          EQ * 8
sum                                             sum
ProTWo                                          ProdTwo
EQ * 8                                          EQ * 8
sum                                             sum
EQ * 6                                          EQ * 6
sum * 2                                         sum * 2
ProdTwo                                         ProdTwo
EQ * 4                                          EQ * 4
sum                                             sum
ProdTwo * 4                                     ProdTwo * 4