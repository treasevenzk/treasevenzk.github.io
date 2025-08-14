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


python
63+23+280+539+471+478+619+355+1321+393+281+596+837+450+281=6987
26+201+66+915+747+147+200+1255+741=4298

```
编译流程
[  0%] Creating directories for 'project_libbacktrace'
[  0%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/int_set.cc.o
[  0%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/canonical_simplify.cc.o
[  0%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/const_int_bound.cc.o
[  0%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/detect_linear_equation.cc.o
[  1%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/domain_touched.cc.o
[  1%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/ir_mutator_with_analyzer.cc.o
[  2%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/iter_affine_map.cc.o
[  2%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/modular_set.cc.o
[  2%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/rewrite_simplify.cc.o
[  2%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/solve_linear_equation.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/solve_linear_inequality.cc.o
[  4%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/bound_deducer.cc.o
[  4%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/compute_dag.cc.o
[  4%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/cost_model.cc.o
[  4%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/int_constraints.cc.o
[  5%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/feature.cc.o
[  5%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/feature_pam.cc.o
[  5%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/analyzer.cc.o
[  5%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/auto_schedule.cc.o
[  7%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/feature_psa.cc.o
[  8%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/loop_state.cc.o
[  8%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/measure.cc.o
[  8%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/measure_record.cc.o
[  8%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_policy/empty_policy.cc.o
[  9%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_policy/search_policy.cc.o
[  9%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_policy/sketch_policy.cc.o
[  9%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_policy/sketch_policy_rules.cc.o
[  9%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_policy/utils.cc.o
[ 10%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_task.cc.o
[ 10%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/transform_step.cc.o
[ 10%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/utils.cc.o
[ 10%] Building CXX object CMakeFiles/tvm_objs.dir/src/autotvm/feature_visitor.cc.o
[ 11%] Building CXX object CMakeFiles/tvm_objs.dir/src/autotvm/touch_extractor.cc.o
[ 11%] Building CXX object CMakeFiles/tvm_objs.dir/src/driver/driver_api.cc.o
[ 11%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/adt.cc.o
[ 13%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/attrs.cc.o
[ 14%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/diagnostic.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/env_func.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/error.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/expr.cc.o
[ 16%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/function.cc.o
[ 16%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/module.cc.o
[ 16%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/op.cc.o
[ 16%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/span.cc.o
[ 18%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/tensor_type.cc.o
[ 18%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/transform.cc.o
[ 18%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/type.cc.o
[ 18%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/type_functor.cc.o
[ 19%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/type_relation.cc.o
[ 19%] Building CXX object CMakeFiles/tvm_objs.dir/src/node/reflection.cc.o
[ 19%] Building CXX object CMakeFiles/tvm_objs.dir/src/node/repr_printer.cc.o
[ 20%] Building CXX object CMakeFiles/tvm_objs.dir/src/node/serialization.cc.o
[ 20%] Building CXX object CMakeFiles/tvm_objs.dir/src/node/structural_equal.cc.o
[ 21%] Building CXX object CMakeFiles/tvm_objs.dir/src/node/structural_hash.cc.o
[ 21%] Building CXX object CMakeFiles/tvm_objs.dir/src/parser/meta_ref.cc.o
[ 22%] Building CXX object CMakeFiles/tvm_objs.dir/src/parser/parser.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_objs.dir/src/parser/source_map.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_objs.dir/src/parser/span_check.cc.o
[ 24%] Building CXX object CMakeFiles/tvm_objs.dir/src/printer/doc.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_objs.dir/src/printer/text_printer.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_objs.dir/src/printer/relay_text_printer.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_objs.dir/src/printer/tir_text_printer.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_objs.dir/src/printer/tvmscript_printer.cc.o
[ 28%] Building CXX object CMakeFiles/tvm_objs.dir/src/support/ffi_testing.cc.o
[ 28%] Building CXX object CMakeFiles/tvm_objs.dir/src/support/hexdump.cc.o
[ 28%] Building CXX object CMakeFiles/tvm_objs.dir/src/support/libinfo.cc.o
[ 29%] Building CXX object CMakeFiles/tvm_objs.dir/src/support/parallel_for.cc.o
[ 30%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/autodiff/ad_simplify.cc.o
[ 30%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/autodiff/ad_utils.cc.o
[ 30%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/autodiff/adjoint.cc.o
[ 30%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/autodiff/jacobian.cc.o
[ 32%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/compute_op.cc.o
[ 32%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/cross_thread_reduction.cc.o
[ 32%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/extern_op.cc.o
[ 32%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/hybrid_op.cc.o
[ 33%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/op_utils.cc.o
[ 33%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/placeholder_op.cc.o
[ 33%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/scan_op.cc.o
[ 33%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/tensor_compute_op.cc.o
[ 34%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/tensorize.cc.o
[ 34%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/auto_inline_elem_wise.cc.o
[ 34%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/bound.cc.o
[ 35%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/graph.cc.o
[ 35%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/message_passing.cc.o
[ 35%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/operation_inline.cc.o
[ 35%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/schedule_dataflow_rewrite.cc.o
[ 36%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/schedule_lang.cc.o
[ 36%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/schedule_ops.cc.o
[ 36%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/schedule_postproc_rewrite_for_tensor_core.cc.o
[ 36%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/schedule_postproc_to_primfunc.cc.o
[ 37%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/verify_compact_buffer.cc.o
[ 37%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/tensor.cc.o
[ 37%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/deep_equal.cc.o
[ 37%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/side_effect.cc.o
[ 38%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/var_touch.cc.o
[ 38%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/verify_gpu_code.cc.o
[ 38%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/verify_memory.cc.o
[ 38%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/verify_ssa.cc.o
[ 39%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/buffer.cc.o
[ 39%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/data_layout.cc.o
[ 39%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/expr.cc.o
[ 39%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/expr_functor.cc.o
[ 40%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/function.cc.o
[ 40%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/stmt.cc.o
[ 40%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/stmt_functor.cc.o
[ 40%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/transform.cc.o
[ 41%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/op/builtin.cc.o
[ 41%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/op/op.cc.o
[ 41%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/op/runtime.cc.o
[ 42%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/arg_binder.cc.o
[ 42%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/bf16_legalize.cc.o
[ 42%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/bound_checker.cc.o
[ 42%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/combine_context_call.cc.o
[ 43%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/coproc_sync.cc.o
[ 43%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/decorate_device_scope.cc.o
[ 43%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/hoist_if_then_else.cc.o
[ 43%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/inject_copy_intrin.cc.o
[ 44%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/inject_double_buffer.cc.o
[ 44%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/inject_prefetch.cc.o
[ 44%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/inject_virtual_thread.cc.o
[ 44%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/ir_utils.cc.o
[ 45%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lift_attr_scope.cc.o
[ 45%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/loop_partition.cc.o
[ 45%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_custom_datatypes.cc.o
[ 45%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_device_storage_access_info.cc.o
[ 46%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_intrin.cc.o
[ 46%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_thread_allreduce.cc.o
[ 46%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_tvm_builtin.cc.o
[ 46%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_warp_memory.cc.o
[ 47%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/make_packed_api.cc.o
[ 47%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/narrow_datatype.cc.o
[ 47%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/remap_thread_axis.cc.o
[ 47%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/remove_no_op.cc.o
[ 48%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/rewrite_unsafe_select.cc.o
[ 48%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/simplify.cc.o
[ 48%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/skip_assert.cc.o
[ 48%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/split_host_device.cc.o
[ 49%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/storage_access.cc.o
[ 49%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/storage_flatten.cc.o
[ 49%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/storage_rewrite.cc.o
[ 50%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/tensorcore_infer_fragment.cc.o
[ 50%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/thread_storage_sync.cc.o
[ 50%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/unroll_loop.cc.o
[ 50%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/vectorize_loop.cc.o
[ 51%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/broadcast.cc.o
[ 51%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/elemwise.cc.o
[ 51%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/nn.cc.o
[ 51%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/reduction.cc.o
[ 52%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/schedule.cc.o
[ 52%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/transform.cc.o
[ 52%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/vision.cc.o
[ 52%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/codegen.cc.o
[ 53%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/func_registry_generator.cc.o
[ 53%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/generic_func.cc.o
[ 53%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/intrin_rule.cc.o
[ 53%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/metadata_module.cc.o
[ 54%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_aocl.cc.o
[ 54%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_c.cc.o
[ 54%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_c_host.cc.o
[ 54%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_cuda.cc.o
[ 55%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_metal.cc.o
[ 55%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_opencl.cc.o
[ 55%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_params.cc.o
[ 55%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_source_base.cc.o
[ 56%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_vhls.cc.o
[ 56%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/intrin_rule_aocl.cc.o
[ 56%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/intrin_rule_cuda.cc.o
[ 56%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/intrin_rule_metal.cc.o
[ 57%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/intrin_rule_opencl.cc.o
[ 57%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/intrin_rule_vhls.cc.o
[ 57%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/source_module.cc.o
[ 58%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/tag.cc.o
[ 58%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/target.cc.o
[ 58%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/target_info.cc.o
[ 58%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/target_kind.cc.o
[ 59%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/algorithm/argsort.cc.o
[ 59%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/algorithm/sort.cc.o
[ 59%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/algorithm/topk.cc.o
[ 59%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/annotation/annotation.cc.o
[ 60%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/debug.cc.o
[ 60%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/dyn/algorithm/topk.cc.o
[ 60%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/dyn/image/resize.cc.o
[ 60%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/dyn/nn/pad.cc.o
[ 61%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/dyn/nn/upsampling.cc.o
[ 61%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/dyn/tensor/transform.cc.o
[ 61%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/image/dilation2d.cc.o
[ 61%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/image/grid_sample.cc.o
[ 62%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/image/resize.cc.o
[ 62%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/memory/memory.cc.o
[ 62%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/bitserial.cc.o
[ 62%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/convolution.cc.o
[ 63%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/correlation.cc.o
[ 63%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/nn.cc.o
[ 63%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/pad.cc.o
[ 63%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/pooling.cc.o
[ 64%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/sparse.cc.o
[ 64%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/upsampling.cc.o
[ 64%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/random/kernel.cc.o
[ 65%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/tensor/binary.cc.o
[ 65%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/tensor/reduce.cc.o
[ 65%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/tensor/transform.cc.o
[ 65%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/tensor/unary.cc.o
[ 66%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/type_relations.cc.o
[ 66%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/vision/multibox_op.cc.o
[ 66%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/vision/nms.cc.o
[ 66%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/vision/rcnn_op.cc.o
[ 67%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/vision/yolo.cc.o
[ 67%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/vm/vm.cc.o
[ 67%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/annotated_region_set.cc.o
[ 67%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/call_graph.cc.o
[ 68%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/context_analysis.cc.o
[ 68%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/dependency_graph.cc.o
[ 68%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/extract_fused_functions.cc.o
[ 68%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/feature.cc.o
[ 69%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/get_calibration_data.cc.o
[ 69%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/kind_check.cc.o
[ 69%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/mac_count.cc.o
[ 69%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/match_exhaustion.cc.o
[ 70%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/type_solver.cc.o
[ 70%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/util.cc.o
[ 70%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/well_formed.cc.o
[ 70%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/quantize/annotate.cc.o
[ 71%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/quantize/calibrate.cc.o
[ 71%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/quantize/partition.cc.o
[ 71%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/quantize/quantize.cc.o
[ 71%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/quantize/realize.cc.o
[ 72%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/alter_op_layout.cc.o
[ 72%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/annotate_target.cc.o
[ 72%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/auto_scheduler_layout_rewrite.cc.o
[ 73%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/canonicalize_cast.cc.o
[ 73%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/canonicalize_ops.cc.o
[ 73%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/combine_parallel_batch_matmul.cc.o
[ 73%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/combine_parallel_conv2d.cc.o
[ 74%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/combine_parallel_dense.cc.o
[ 74%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/combine_parallel_op.cc.o
[ 74%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/combine_parallel_op_batch.cc.o
[ 74%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/convert_layout.cc.o
[ 75%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/convert_sparse_dense.cc.o
[ 75%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/de_duplicate.cc.o
[ 75%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/dead_code.cc.o
[ 75%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/defunctionalization.cc.o
[ 76%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/defuse_ops.cc.o
[ 76%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/device_annotation.cc.o
[ 76%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/dynamic_to_static.cc.o
[ 76%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/eliminate_common_subexpr.cc.o
[ 77%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/eta_expand.cc.o
[ 77%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/expr_subst.cc.o
[ 77%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/fast_math.cc.o
[ 77%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/first_order_gradient.cc.o
[ 78%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/fold_constant.cc.o
[ 78%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/fold_explicit_padding.cc.o
[ 78%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/fold_scale_axis.cc.o
[ 78%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/forward_rewrite.cc.o
[ 79%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/fuse_ops.cc.o
[ 79%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/higher_order_gradient.cc.o
[ 79%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/inline.cc.o
[ 79%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/lazy_gradient_init.cc.o
[ 80%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/legalize.cc.o
[ 80%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/memory_alloc.cc.o
[ 80%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/merge_compiler_regions.cc.o
[ 81%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/merge_composite.cc.o
[ 81%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/partial_eval.cc.o
[ 81%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/partition_graph.cc.o
[ 81%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/simplify_expr.cc.o
[ 82%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/simplify_fc_transpose.cc.o
[ 82%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/simplify_inference.cc.o
[ 82%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/to_a_normal_form.cc.o
[ 82%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/to_basic_block_normal_form.cc.o
[ 83%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/to_cps.cc.o
[ 83%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/to_graph_normal_form.cc.o
[ 83%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/type_infer.cc.o
[ 83%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/build_module.cc.o
[ 84%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/compile_engine.cc.o
[ 84%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/graph_plan_memory.cc.o
[ 84%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/graph_runtime_codegen.cc.o
[ 84%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/interpreter.cc.o
[ 85%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/param_dict.cc.o
[ 85%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/vm/compiler.cc.o
[ 85%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/vm/inline_primitives.cc.o
[ 85%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/vm/lambda_lift.cc.o
[ 86%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/vm/removed_unused_funcs.cc.o
[ 86%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/adt.cc.o
[ 86%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/base.cc.o
[ 86%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/dataflow_matcher.cc.o
[ 87%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/dataflow_pattern.cc.o
[ 87%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/dataflow_pattern_functor.cc.o
[ 87%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/expr.cc.o
[ 88%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/expr_functor.cc.o
[ 88%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/function.cc.o
[ 88%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/indexed_graph.cc.o
[ 88%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/op_strategy.cc.o
[ 89%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/pattern_functor.cc.o
[ 89%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/transform.cc.o
[ 89%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/add.cc.o
[ 89%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/concatenate.cc.o
[ 90%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/convolution.cc.o
[ 90%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/convolution_transpose.cc.o
[ 90%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/dense.cc.o
[ 90%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/dequantize.cc.o
[ 91%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/mul.cc.o
[ 91%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/quantize.cc.o
[ 91%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/requantize.cc.o
[ 91%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/simulated_dequantize.cc.o
[ 92%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/simulated_quantize.cc.o
[ 92%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/subtract.cc.o
[ 92%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/pass/legalize.cc.o
[ 92%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/utils.cc.o
[ 93%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/datatype/registry.cc.o
[ 93%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/datatype/myfloat/myfloat.cc.o
[ 93%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/stackvm/codegen_stackvm.cc.o
[ 93%] Building CXX object CMakeFiles/tvm_objs.dir/src/runtime/stackvm/stackvm.cc.o
[ 94%] Building CXX object CMakeFiles/tvm_objs.dir/src/runtime/stackvm/stackvm_module.cc.o
[ 94%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_cuda_on.cc.o
[ 94%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_hexagon_off.cc.o
[ 94%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_sdaccel_off.cc.o
[ 95%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_aocl_off.cc.o
[ 95%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_opencl_off.cc.o
[ 95%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_metal_off.cc.o
[ 96%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_rocm_off.cc.o
[ 96%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_amdgpu.cc.o
[ 96%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_arm.cc.o
[ 96%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_blob.cc.o
[ 97%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_cpu.cc.o
[ 97%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_hexagon.cc.o
[ 97%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_llvm.cc.o
[ 97%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_nvptx.cc.o
[ 98%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_params.cc.o
[ 98%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_x86_64.cc.o
[ 98%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/intrin_rule_hexagon.cc.o
[ 98%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/intrin_rule_llvm.cc.o
[ 99%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/intrin_rule_nvptx.cc.o
[ 99%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/intrin_rule_rocm.cc.o
[ 99%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/llvm_common.cc.o
[ 99%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/llvm_module.cc.o
[100%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/contrib/codegen_c/codegen.cc.o
[100%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/hybrid/codegen_hybrid.cc.o
[100%] Built target tvm_objs
[100%] Linking CXX shared library libtvm.so
[100%] Built target tvm



[ 11%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/builtin_fp16.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/c_runtime_api.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/container.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/cpu_device_api.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/dso_library.cc.o
[ 13%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/file_utils.cc.o
[ 13%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/library_module.cc.o
[ 13%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/logging.cc.o
[ 14%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/metadata_module.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/module.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/ndarray.cc.o
[ 16%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/object.cc.o
[ 17%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/profiling.cc.o
[ 17%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/registry.cc.o
[ 18%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/system_library.cc.o
[ 18%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/thread_pool.cc.o
[ 21%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/threading_backend.cc.o
[ 21%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/bytecode.cc.o
[ 21%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/executable.cc.o
[ 21%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/memory_manager.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/vm.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/workspace_pool.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_channel.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_device_api.cc.o
[ 24%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_endpoint.cc.o
[ 24%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_event_impl.cc.o
[ 24%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_local_session.cc.o
[ 24%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_module.cc.o
[ 25%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_pipe_impl.cc.o
[ 25%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_server_env.cc.o
[ 25%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_session.cc.o
[ 25%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_socket_impl.cc.o
[ 26%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph/graph_runtime.cc.o
[ 26%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph/graph_runtime_factory.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph/debug/graph_runtime_debug.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/profiler/vm.cc.o
[ 29%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/cuda/cuda_device_api.cc.o
[ 29%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/cuda/cuda_module.cc.o
[ 30%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/contrib/random/random.cc.o
[ 31%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/contrib/sort/sort.cc.o
```


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

