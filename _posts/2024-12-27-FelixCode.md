---
layout:     post
title:      FelixCode
subtitle:   Code Reproduction
date:       2024-12-27
author:     Treaseven
header-img: img/bg20.jpg
catalog: true
tags:
    - debug
---



修改部分的内容

-----------------------------
include

arith
egg_simpl.h、var_context.h

tir
op.h、stmt_functor.h、var.h


auto_scheduler
compute_dag.h、loop_state.h、transform_step.h

driver
driver_api.h

te
schedule_pass.h

topi/nn
pooling.h

----------------------------
src

auto_scheduler
compute_dag.cc、transform_step.cc、utils.h、loop_state.cc、auto_schedule.cc、feature.cc、measure.cc、measure_record.cc、utils.cc
search_policy: empty_policy.cc、search_policy.cc、sketch_policy.cc、sketch_policy_rules.cc、utils.cc

arith
egg_simpl/src   lang.rs、lib.rs
egg_simpl.cc、var_context.cc

tir/op
op.cc

tir/ir
stmt_functor.cc、expr.cc

tir/transforms
inject_virtual_thread.cc
vectorize_loop.cc

driver
driver_api.cc

te/schedule
bound.cc、message_passing.cc、message_passing.h

te/operation
op_utils.cc


felix
sketch_rules.cc、sketch_rules.h、utils.cc、utils.h、constraints.cc、feat_transform.cc、features.cc、features.h、rangeinfer.h

-----------------------------
python

auto_scheduler
compute_dag.py、relay_integration.py、task_scheduler.py
cost_model: __init__.py、dataset.py、metric.py、mlp_model.py、xgb_model.py

te/hybrid
parser.py

te
operation.py

topi
math.py
nn: elemwise.py、batch_matmul.py、conv2d.py、conv2d_transpose.py
cuda: conv2d_transpose.py、conv2d_nhwc_tensorcore.py、conv2d_nhwc_winograd.py

tir
expr.py

felix
__init__.py、ffi.py、sym_dag.py、sym_task.py、utils.py、logging.py、sketch.py、features.py、optim.py、cost_model.py、_ffi_api.py、ansor_tune.py、test_extract_task.py
nn:__init__.py、dcgan.py、vit.py、llama.py

relay/backend
te_compiler.py

relay/op/strategy
cuda.py

relay/frontend
pytorch.py

scripts
felix_cost_model.py、patch_tenset_dataset.py、ansor_cost_model.py、ansor_tune_network.py、train_cost_model.py、tenset_measure_programs.py、tf_torch_perf.py、felix_tune_network.py
ftf_nns: __init__.py、dcgan.py、r3d.py、vit.py


