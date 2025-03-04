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





