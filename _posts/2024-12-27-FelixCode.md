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
```
-----------------------------
include

arith
egg_simpl.h(+)、var_context.h(+)

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
egg_simpl/src   lang.rs(+)、lib.rs(+)
egg_simpl.cc(+)、var_context.cc(+)

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

felix(+)
sketch_rules.cc、sketch_rules.h、utils.cc、utils.h、constraints.cc、feat_transform.cc、features.cc、features.h、rangeinfer.h

-----------------------------
python

auto_scheduler
compute_dag.py、relay_integration.py、task_scheduler.py
cost_model: __init__.py、dataset.py(+)、metric.py(+)、mlp_model.py(+)、xgb_model.py

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

felix(+)
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
tf_nns: __init__.py、dcgan.py、r3d.py、vit.py
```

from pathlib import Path
Path处理文件路径

lr_scheduler.MultiStepLR 是Pytorch中用于动态调整学习率的工具，其作用是根据预设的里程牌分阶段降低学习率
lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
optimizer:绑定的优化器对象(如SGD、Adam)     milestones:预设的epoch节点列表(需严格递增)      gamma:衰减系数，每次调整时学习率乘以该系数

@dataclass装饰器：用于自动生成特殊方法如__init__、__repr__、__eq__等
类变量、实例变量：类变量是所有的实例都共享，实例变量定义在dataclass装饰器下，会被自动添加到__init__方法中
@classmethod装饰器：用于定义类方法而不是实例方法，类方法的第一个参数通常命名为cls(而不是self),表示类本身，类方法可以通过类名直接调用，而不需要先创建类的实例
@staticmethod： 将方法定义为静态方法，不需要实例化类即可调用
@property： 将方法转换为属性，使其可以像属性一样访问
@abc.abstractmethod： 用于在抽象基类中定义抽象方法，标记一个方法为抽象方法，意味着该方法必须在子类中实现，如果子类没有实现所有的抽象方法，那么尝试实例化改子类时会引发TypeError异常

ctypes.byref()是python中的ctypes库中的一个函数，用于高效地传递C函数参数中所需的指针

ast.literal_eval： 安全地将字符串形式的Python字面值转换为相应的Python对象

setattr()函数的作用是设置对象的属性值，接受三个参数：对象(要设置属性的对象)、属性名(要设置的属性名称)、属性值(要为属性设置的值)
setattr(object, name, value)
any() 接受一个可迭代对象，如果可迭代对象中任意一个元素为True，则返回True，否则返回False

extern "C" {} :解决C++与C之间的符号链接兼容性问题，强制C++以C的规则编译函数，禁止名称修饰

#ifndef/#else/#endif :实现条件编译，根据宏定义决定代码是否参与编译，#ifndef MACRO: 如果未定义MACRO，编译后续代码； #else: 否则编译另一段代码； #endif ：结束条件编译块

362+235+546+70=1213
207+1072+1072+30+263+640+100+280+81=3745
92+201+308+183+181+357+123+73+425+1617+307+409=4276


from tvm import felix经历过程
import tvm._ffi
模块加载，加载python/tvm/_ffi/__init__.py文件；
基础设施准备：(加载base.py：初始化与C++库的基本连接，加载C++动态库通过_load_lib()函数)、加载registry.py：设置各种注册函数和类型转换机制）；
C++库加载：在base.py中，_load_lib()函数查找并加载TVM的C++动态库，库加载之后，设置全局变量_LIB指向这个库，使Python可以调用C++函数
tvm._ffi._init_api("felix", __name__)
全局函数查找，调用list_global_func_names()函数，获取C++端所有注册的全局函数名称
函数筛选与导入，使用get_global_func获取对应的函数对象



felix的src内容：
注册的全局函数felix： GetFeaturePerk、LinearExprAsPrimExpr、GenerateAllSymSketches、ExtractBackbone、PrintTrSteop、GenerateCodeForState、ExtractConfigDict、StateFromConfig、MeasurePerformance




编译流程
```
[  0%] Generating ../../egg_simpl/release/libegg_simpl.a, ../../egg_simpl/release/egg_simpl.h
[  1%] Building CXX object CMakeFiles/tvm_libinfo_objs.dir/src/support/libinfo.cc.o
[  1%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/const_int_bound.cc.o
[  1%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/bound_deducer.cc.o
[  1%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/detect_linear_equation.cc.o
[  1%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/canonical_simplify.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/int_constraints.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/modular_set.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/domain_touched.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/int_set.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/ir_mutator_with_analyzer.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/analyzer.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/solve_linear_equation.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/rewrite_simplify.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/solve_linear_inequality.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/auto_schedule.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/egg_simpl.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/iter_affine_map.cc.o
[  3%] Building CXX object CMakeFiles/tvm_objs.dir/src/arith/var_context.cc.o
[  5%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/compute_dag.cc.o
[  5%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/cost_model.cc.o
[  5%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/feature.cc.o
[  5%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/loop_state.cc.o
[  5%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/measure.cc.o
[  5%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/measure_record.cc.o
[  6%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_policy/empty_policy.cc.o
[  6%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_policy/search_policy.cc.o
[  6%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_policy/sketch_policy.cc.o
[  6%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_policy/sketch_policy_rules.cc.o
[  6%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_policy/utils.cc.o
[  7%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/search_task.cc.o
[  7%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/transform_step.cc.o
[  7%] Building CXX object CMakeFiles/tvm_objs.dir/src/auto_scheduler/utils.cc.o                          
[  7%] Building CXX object CMakeFiles/tvm_objs.dir/src/autotvm/feature_visitor.cc.o
[  7%] Building CXX object CMakeFiles/tvm_objs.dir/src/autotvm/touch_extractor.cc.o
[  7%] Building CXX object CMakeFiles/tvm_objs.dir/src/driver/driver_api.cc.o
[  8%] Building CXX object CMakeFiles/tvm_objs.dir/src/felix/constraints.cc.o
[  8%] Building CXX object CMakeFiles/tvm_objs.dir/src/felix/feat_transform.cc.o
[  8%] Building CXX object CMakeFiles/tvm_objs.dir/src/felix/features.cc.o
[  8%] Building CXX object CMakeFiles/tvm_objs.dir/src/felix/sketch_rules.cc.o
[  8%] Building CXX object CMakeFiles/tvm_objs.dir/src/felix/utils.cc.o
[  9%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/adt.cc.o
[ 10%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/affine_type.cc.o
[ 11%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/attrs.cc.o
[ 11%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/diagnostic.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/env_func.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/error.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/expr.cc.o
[ 13%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/function.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/instrument.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/module.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/op.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/span.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/tensor_type.cc.o
[ 16%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/transform.cc.o
[ 16%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/type.cc.o
[ 17%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/type_functor.cc.o
[ 17%] Building CXX object CMakeFiles/tvm_objs.dir/src/ir/type_relation.cc.o
[ 17%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/arg_info.cc.o
[ 17%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/builder/builder.cc.o
[ 18%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/cost_model/cost_model.cc.o
[ 18%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/database/database.cc.o
[ 18%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/database/json_database.cc.o
[ 19%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/feature_extractor/feature_extractor.cc.o
[ 20%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/feature_extractor/per_store_feature.cc.o
[ 21%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/integration.cc.o
[ 22%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/measure_callback/add_to_database.cc.o
[ 22%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/measure_callback/echo_statistics.cc.o
[ 22%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/measure_callback/measure_callback.cc.o
[ 22%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/measure_callback/remove_build_artifact.cc.o
[ 22%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/measure_callback/update_cost_model.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/mutator/mutate_compute_location.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/mutator/mutate_parallel.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/mutator/mutate_tile_size.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/mutator/mutate_unroll.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/mutator/mutator.cc.o
[ 23%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/postproc/disallow_dynamic_loop.cc.o
[ 24%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/postproc/postproc.cc.o
[ 24%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/postproc/rewrite_cooperative_fetch.cc.o
[ 24%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/postproc/rewrite_parallel_vectorize_unroll.cc.o
[ 24%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/postproc/rewrite_reduction_block.cc.o
[ 24%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/postproc/rewrite_unbound_block.cc.o
[ 24%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/postproc/verify_gpu_code.cc.o
[ 25%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/runner/runner.cc.o
[ 25%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/schedule_rule/add_rfactor.cc.o
[ 25%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/schedule_rule/auto_inline.cc.o
[ 25%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/schedule_rule/cross_thread_reduction.cc.o
[ 25%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/schedule_rule/multi_level_tiling.cc.o
[ 25%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/schedule_rule/parallel_vectorize_unroll.cc.o
[ 26%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/schedule_rule/random_compute_location.cc.o
[ 26%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/schedule_rule/schedule_rule.cc.o
[ 26%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/search_strategy/evolutionary_search.cc.o
[ 26%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/search_strategy/replay_func.cc.o
[ 26%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/search_strategy/replay_trace.cc.o
[ 26%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/search_strategy/search_strategy.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/space_generator/post_order_apply.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/space_generator/space_generator.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/space_generator/space_generator_union.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/task_scheduler/round_robin.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/task_scheduler/task_scheduler.cc.o
[ 27%] Building CXX object CMakeFiles/tvm_objs.dir/src/meta_schedule/tune_context.cc.o
[ 29%] Building CXX object CMakeFiles/tvm_objs.dir/src/node/container_printing.cc.o
[ 29%] Building CXX object CMakeFiles/tvm_objs.dir/src/node/reflection.cc.o
[ 29%] Building CXX object CMakeFiles/tvm_objs.dir/src/node/repr_printer.cc.o
[ 29%] Building CXX object CMakeFiles/tvm_objs.dir/src/node/serialization.cc.o
[ 29%] Building CXX object CMakeFiles/tvm_objs.dir/src/node/structural_equal.cc.o
[ 30%] Building CXX object CMakeFiles/tvm_objs.dir/src/node/structural_hash.cc.o
[ 30%] Building CXX object CMakeFiles/tvm_objs.dir/src/parser/meta_ref.cc.o
[ 30%] Building CXX object CMakeFiles/tvm_objs.dir/src/parser/parser.cc.o
[ 30%] Building CXX object CMakeFiles/tvm_objs.dir/src/parser/source_map.cc.o
[ 30%] Building CXX object CMakeFiles/tvm_objs.dir/src/parser/span_check.cc.o
[ 30%] Building CXX object CMakeFiles/tvm_objs.dir/src/printer/doc.cc.o
[ 31%] Building CXX object CMakeFiles/tvm_objs.dir/src/printer/model_library_format_printer.cc.o
[ 31%] Building CXX object CMakeFiles/tvm_objs.dir/src/printer/relay_text_printer.cc.o
[ 31%] Building CXX object CMakeFiles/tvm_objs.dir/src/printer/text_printer.cc.o
[ 31%] Building CXX object CMakeFiles/tvm_objs.dir/src/printer/tvmscript_printer.cc.o
[ 31%] Building CXX object CMakeFiles/tvm_objs.dir/src/printer/tir_text_printer.cc.o
[ 31%] Building CXX object CMakeFiles/tvm_objs.dir/src/support/ffi_testing.cc.o
[ 32%] Building CXX object CMakeFiles/tvm_objs.dir/src/support/hexdump.cc.o
[ 32%] Building CXX object CMakeFiles/tvm_objs.dir/src/support/parallel_for.cc.o
[ 32%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/autodiff/ad_simplify.cc.o
[ 32%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/autodiff/ad_utils.cc.o
[ 32%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/autodiff/adjoint.cc.o
[ 32%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/autodiff/jacobian.cc.o
[ 33%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/compute_op.cc.o
[ 33%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/create_primfunc.cc.o
[ 33%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/cross_thread_reduction.cc.o
[ 33%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/extern_op.cc.o
[ 33%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/hybrid_op.cc.o
[ 33%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/op_utils.cc.o
[ 34%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/placeholder_op.cc.o
[ 34%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/scan_op.cc.o
[ 34%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/tensor_compute_op.cc.o
[ 34%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/operation/tensorize.cc.o
[ 34%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/auto_inline_elem_wise.cc.o
[ 34%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/bound.cc.o
[ 35%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/graph.cc.o
[ 35%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/message_passing.cc.o
[ 35%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/operation_inline.cc.o
[ 35%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/schedule_dataflow_rewrite.cc.o
[ 35%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/schedule_lang.cc.o
[ 35%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/schedule_ops.cc.o
[ 36%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/schedule_postproc_to_primfunc.cc.o
[ 36%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/schedule/verify_compact_buffer.cc.o
[ 36%] Building CXX object CMakeFiles/tvm_objs.dir/src/te/tensor.cc.o
[ 36%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/block_access_region_detector.cc.o
[ 36%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/buffer_access_lca_detector.cc.o
[ 37%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/calculate_workspace.cc.o
[ 37%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/deep_equal.cc.o
[ 37%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/device_constraint_utils.cc.o
[ 37%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/expr_complexity.cc.o
[ 37%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/side_effect.cc.o
[ 37%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/var_touch.cc.o
[ 38%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/verify_gpu_code.cc.o
[ 38%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/verify_memory.cc.o
[ 38%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/analysis/verify_ssa.cc.o
[ 38%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/buffer.cc.o
[ 38%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/data_layout.cc.o
[ 38%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/expr.cc.o
[ 39%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/expr_functor.cc.o
[ 39%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/function.cc.o
[ 39%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/script/script_complete.cc.o
[ 39%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/specialize.cc.o
[ 39%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/stmt.cc.o
[ 39%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/stmt_functor.cc.o
[ 40%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/ir/transform.cc.o
[ 40%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/op/builtin.cc.o
[ 40%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/op/op.cc.o
[ 40%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/op/runtime.cc.o
[ 40%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/analysis/analysis.cc.o
[ 40%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/analysis/verify.cc.o
[ 41%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/block_scope.cc.o
[ 41%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/concrete_schedule.cc.o
[ 41%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/error.cc.o
[ 41%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/instruction.cc.o
[ 41%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/ir_comparator.cc.o
[ 41%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/primitive/annotate.cc.o
[ 43%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/primitive/block_annotate.cc.o
[ 43%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/primitive/blockize_tensorize.cc.o
[ 43%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/primitive/cache_read_write.cc.o
[ 43%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/primitive/compute_at.cc.o
[ 43%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/primitive/compute_inline.cc.o
[ 44%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/primitive/for_kind.cc.o
[ 44%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/primitive/get_block_loop.cc.o
[ 44%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/primitive/loop_transformation.cc.o
[ 44%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/primitive/reduction.cc.o
[ 44%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/primitive/sampling.cc.o
[ 44%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/schedule.cc.o
[ 45%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/state.cc.o
[ 45%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/trace.cc.o
[ 45%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/traced_schedule.cc.o
[ 45%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/schedule/transform.cc.o
[ 45%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/arg_binder.cc.o
[ 45%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/bf16_legalize.cc.o
[ 46%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/bound_checker.cc.o
[ 46%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/combine_context_call.cc.o
[ 46%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/compact_buffer_region.cc.o
[ 46%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/convert_blocks_to_opaque.cc.o
[ 46%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/convert_for_loops_serial.cc.o
[ 46%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/coproc_sync.cc.o
[ 47%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/decorate_device_scope.cc.o
[ 47%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/flatten_buffer.cc.o
[ 47%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/hoist_if_then_else.cc.o
[ 47%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/inject_copy_intrin.cc.o
[ 47%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/inject_double_buffer.cc.o
[ 47%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/inject_prefetch.cc.o
[ 48%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/inject_rolling_buffer.cc.o
[ 48%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/inject_virtual_thread.cc.o
[ 48%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/ir_utils.cc.o
[ 48%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/legalize_packed_calls.cc.o
[ 48%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lift_attr_scope.cc.o
[ 48%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/loop_partition.cc.o
[ 49%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_cross_thread_reduction.cc.o
[ 49%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_custom_datatypes.cc.o
[ 49%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_device_storage_access_info.cc.o
[ 49%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_init_block.cc.o
[ 49%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_intrin.cc.o
[ 49%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_match_buffer.cc.o
[ 50%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_thread_allreduce.cc.o
[ 50%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_tvm_builtin.cc.o
[ 50%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/lower_warp_memory.cc.o
[ 50%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/make_packed_api.cc.o
[ 50%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/make_unpacked_api.cc.o
[ 51%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/merge_dynamic_shared_memory_allocations.cc.o
[ 51%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/narrow_datatype.cc.o
[ 51%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/plan_update_buffer_allocation_location.cc.o
[ 51%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/remap_thread_axis.cc.o
[ 51%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/remove_no_op.cc.o
[ 51%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/rewrite_unsafe_select.cc.o
[ 52%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/simplify.cc.o
[ 52%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/skip_assert.cc.o
[ 52%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/split_host_device.cc.o
[ 52%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/storage_access.cc.o
[ 52%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/storage_flatten.cc.o
[ 52%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/storage_rewrite.cc.o
[ 53%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/tensorcore_infer_fragment.cc.o
[ 53%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/texture_flatten.cc.o
[ 53%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/thread_storage_sync.cc.o
[ 53%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/unify_thread_binding.cc.o
[ 53%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/unroll_loop.cc.o
[ 53%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/update_pointer_storage_scope.cc.o
[ 54%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/transforms/vectorize_loop.cc.o
[ 54%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/usmp/algo/greedy.cc.o
[ 54%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/usmp/algo/hill_climb.cc.o
[ 54%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/usmp/analysis/extract_buffer_info.cc.o
[ 54%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/usmp/transform/assign_pool_info.cc.o
[ 54%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/usmp/transform/convert_pool_allocations_to_offsets.cc.o
[ 55%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/usmp/unified_static_memory_planner.cc.o
[ 55%] Building CXX object CMakeFiles/tvm_objs.dir/src/tir/usmp/utils.cc.o
[ 55%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/broadcast.cc.o
[ 55%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/elemwise.cc.o
[ 55%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/nn.cc.o
[ 55%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/reduction.cc.o
[ 56%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/schedule.cc.o
[ 56%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/transform.cc.o
[ 56%] Building CXX object CMakeFiles/tvm_objs.dir/src/topi/vision.cc.o
[ 56%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/compilation_config.cc.o
[ 56%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/codegen.cc.o
[ 56%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/func_registry_generator.cc.o
[ 58%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/generic_func.cc.o
[ 58%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/intrin_rule.cc.o
[ 58%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/metadata_module.cc.o
[ 58%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_aocl.cc.o
[ 58%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_c.cc.o
[ 59%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_c_host.cc.o
[ 59%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_cuda.cc.o
[ 59%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_metal.cc.o
[ 59%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_opencl.cc.o
[ 59%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_params.cc.o
[ 59%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_source_base.cc.o
[ 60%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/codegen_vhls.cc.o
[ 60%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/interface_c.cc.o
[ 60%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/intrin_rule_aocl.cc.o
[ 60%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/intrin_rule_cuda.cc.o
[ 60%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/intrin_rule_metal.cc.o
[ 60%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/intrin_rule_opencl.cc.o
[ 61%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/intrin_rule_vhls.cc.o
[ 61%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/ptx_mma.cc.o
[ 61%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/source/source_module.cc.o
[ 61%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/tag.cc.o
[ 61%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/target.cc.o
[ 61%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/target_info.cc.o
[ 62%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/target_kind.cc.o
[ 62%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/virtual_device.cc.o
[ 62%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/algorithm/argsort.cc.o
[ 62%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/algorithm/searchsorted.cc.o
[ 62%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/algorithm/sort.cc.o
[ 62%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/algorithm/topk.cc.o
[ 63%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/annotation/annotation.cc.o
[ 63%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/call/call.cc.o
[ 63%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/contrib/ethosu/binary_elementwise.cc.o
[ 63%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/contrib/ethosu/common.cc.o
[ 63%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/contrib/ethosu/convolution.cc.o
[ 63%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/contrib/ethosu/depthwise.cc.o
[ 64%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/contrib/ethosu/identity.cc.o
[ 64%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/contrib/ethosu/pooling.cc.o
[ 64%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/contrib/ethosu/unary_elementwise.cc.o
[ 64%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/debug.cc.o
[ 64%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/dyn/algorithm/topk.cc.o
[ 65%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/dyn/image/resize.cc.o
[ 65%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/dyn/nn/pad.cc.o
[ 65%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/dyn/nn/upsampling.cc.o
[ 65%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/dyn/tensor/transform.cc.o
[ 65%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/image/dilation2d.cc.o
[ 65%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/image/grid_sample.cc.o
[ 66%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/image/resize.cc.o
[ 66%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/memory/device_copy.cc.o
[ 66%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/memory/memory.cc.o
[ 66%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/memory/on_device.cc.o
[ 66%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/bitserial.cc.o
[ 66%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/convolution.cc.o
[ 67%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/correlation.cc.o
[ 67%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/nn.cc.o
[ 67%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/pad.cc.o
[ 67%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/pooling.cc.o
[ 67%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/sparse.cc.o
[ 67%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/nn/upsampling.cc.o
[ 68%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/random/kernel.cc.o
[ 68%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/tensor/binary.cc.o
[ 68%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/tensor/math.cc.o
[ 68%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/tensor/reduce.cc.o
[ 68%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/tensor/transform.cc.o
[ 68%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/tensor/unary.cc.o
[ 69%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/type_relations.cc.o
[ 69%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/vision/multibox_op.cc.o
[ 69%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/vision/nms.cc.o
[ 69%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/vision/rcnn_op.cc.o
[ 69%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/vision/yolo.cc.o
[ 69%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/op/vm/vm.cc.o
[ 70%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/annotated_region_set.cc.o
[ 70%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/call_graph.cc.o
[ 70%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/dependency_graph.cc.o
[ 70%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/extract_fake_quantized_ops.cc.o
[ 70%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/extract_fused_functions.cc.o
[ 70%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/extract_operators.cc.o
[ 72%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/feature.cc.o
[ 72%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/get_calibration_data.cc.o
[ 72%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/kind_check.cc.o
[ 72%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/mac_count.cc.o
[ 72%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/match_exhaustion.cc.o
[ 73%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/type_solver.cc.o
[ 73%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/util.cc.o
[ 73%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/analysis/well_formed.cc.o
[ 73%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/quantize/annotate.cc.o
[ 73%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/quantize/calibrate.cc.o
[ 73%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/quantize/partition.cc.o
[ 74%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/quantize/quantize.cc.o
[ 74%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/quantize/realize.cc.o
[ 74%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/alter_op_layout.cc.o
[ 74%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/annotate_target.cc.o
[ 74%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/auto_scheduler_layout_rewrite.cc.o
[ 74%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/canonicalize_cast.cc.o
[ 75%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/canonicalize_ops.cc.o
[ 75%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/combine_parallel_batch_matmul.cc.o
[ 75%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/combine_parallel_conv2d.cc.o
[ 75%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/combine_parallel_dense.cc.o
[ 75%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/combine_parallel_op.cc.o
[ 75%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/combine_parallel_op_batch.cc.o
[ 76%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/convert_layout.cc.o
[ 76%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/convert_sparse_conv2d.cc.o
[ 76%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/convert_sparse_dense.cc.o
[ 76%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/de_duplicate.cc.o
[ 76%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/dead_code.cc.o
[ 76%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/defunctionalization.cc.o
[ 77%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/defuse_ops.cc.o
[ 77%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/device_aware_visitors.cc.o
[ 77%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/device_domains.cc.o
[ 77%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/device_planner.cc.o
[ 77%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/dynamic_to_static.cc.o
[ 77%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/eliminate_common_subexpr.cc.o
[ 78%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/eta_expand.cc.o
[ 78%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/expr_subst.cc.o
[ 78%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/fake_quantization_to_integer.cc.o
[ 78%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/fast_math.cc.o
[ 78%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/first_order_gradient.cc.o
[ 78%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/fold_constant.cc.o
[ 79%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/fold_explicit_padding.cc.o
[ 79%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/fold_scale_axis.cc.o
[ 79%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/forward_rewrite.cc.o
[ 79%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/fuse_ops.cc.o
[ 79%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/higher_order_gradient.cc.o
[ 80%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/inline.cc.o
[ 80%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/label_ops.cc.o
[ 80%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/lazy_gradient_init.cc.o
[ 80%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/legalize.cc.o
[ 80%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/memory_alloc.cc.o
[ 80%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/merge_compiler_regions.cc.o
[ 81%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/merge_composite.cc.o
[ 81%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/partial_eval.cc.o
[ 81%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/partition_graph.cc.o
[ 81%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/simplify_expr.cc.o
[ 81%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/simplify_fc_transpose.cc.o
[ 81%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/simplify_inference.cc.o
[ 82%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/split_args.cc.o
[ 82%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/target_hooks.cc.o
[ 82%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/to_a_normal_form.cc.o
[ 82%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/to_basic_block_normal_form.cc.o
[ 82%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/to_cps.cc.o
[ 82%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/to_graph_normal_form.cc.o
[ 83%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/to_mixed_precision.cc.o
[ 83%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/transforms/type_infer.cc.o
[ 83%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/aot_executor_codegen.cc.o
[ 83%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/build_module.cc.o
[ 83%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/executor.cc.o
[ 83%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/graph_executor_codegen.cc.o
[ 84%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/graph_plan_memory.cc.o
[ 84%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/interpreter.cc.o
[ 84%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/name_transforms.cc.o
[ 84%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/param_dict.cc.o
[ 84%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/runtime.cc.o
[ 84%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/te_compiler.cc.o
[ 86%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/te_compiler_cache.cc.o
[ 86%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/utils.cc.o
[ 86%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/vm/compiler.cc.o
[ 86%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/vm/lambda_lift.cc.o
[ 86%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/vm/removed_unused_funcs.cc.o
[ 87%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/adt.cc.o
[ 87%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/base.cc.o
[ 87%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/dataflow_matcher.cc.o
[ 87%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/dataflow_pattern.cc.o
[ 87%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/dataflow_pattern_functor.cc.o
[ 87%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/expr.cc.o
[ 88%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/expr_functor.cc.o
[ 88%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/function.cc.o
[ 88%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/indexed_graph.cc.o
[ 88%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/op_strategy.cc.o
[ 88%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/pattern_functor.cc.o
[ 88%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/ir/transform.cc.o
[ 89%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/add.cc.o
[ 89%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/batch_matmul.cc.o
[ 89%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/concatenate.cc.o
[ 89%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/convolution.cc.o
[ 89%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/convolution_transpose.cc.o
[ 89%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/dense.cc.o
[ 90%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/dequantize.cc.o
[ 90%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/mul.cc.o
[ 90%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/quantize.cc.o
[ 90%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/requantize.cc.o
[ 90%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/requantize_config.cc.o
[ 90%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/rsqrt.cc.o
[ 91%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/simulated_dequantize.cc.o
[ 91%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/simulated_quantize.cc.o
[ 91%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/op/subtract.cc.o
[ 91%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/pass/legalize.cc.o
[ 91%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/qnn/utils.cc.o
[ 91%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/datatype/registry.cc.o
[ 92%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/datatype/myfloat/myfloat.cc.o
[ 92%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/stackvm/codegen_stackvm.cc.o
[ 92%] Building CXX object CMakeFiles/tvm_objs.dir/src/runtime/stackvm/stackvm.cc.o
[ 92%] Building CXX object CMakeFiles/tvm_objs.dir/src/runtime/stackvm/stackvm_module.cc.o



cmake rules
[ 92%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_cuda_on.cc.o        CUDA.cmake
[ 92%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_hexagon_off.cc.o    Hexagon.cmake
[ 93%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_sdaccel_off.cc.o    OpenCL.cmake
[ 93%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_aocl_off.cc.o       OpenCL.cmake
[ 93%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_opencl_off.cc.o     OpenCL.cmake
[ 93%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_metal_off.cc.o      Meatal.cmake
[ 93%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/opt/build_rocm_off.cc.o       ROCM.cmake
[ 94%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_amdgpu.cc.o
[ 94%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_arm.cc.o
[ 94%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_blob.cc.o
[ 94%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_cpu.cc.o
[ 94%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_hexagon.cc.o
[ 94%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_llvm.cc.o
[ 95%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_nvptx.cc.o
[ 95%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_params.cc.o          LLVM.cmake
[ 95%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/codegen_x86_64.cc.o
[ 95%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/intrin_rule_hexagon.cc.o
[ 95%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/intrin_rule_llvm.cc.o
[ 95%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/intrin_rule_nvptx.cc.o
[ 96%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/intrin_rule_rocm.cc.o
[ 96%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/llvm_common.cc.o
[ 96%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/llvm_module.cc.o
[ 96%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/ethosu/cascader/block_config.cc.o        EthosU.cmake
[ 96%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/ethosu/cascader/cascader_options.cc.o
[ 96%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/ethosu/cascader/graph.cc.o
[ 97%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/ethosu/cascader/pareto.cc.o
[ 97%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/ethosu/cascader/parts/ethosu.cc.o
[ 97%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/ethosu/cascader/parts/inline.cc.o
[ 97%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/ethosu/cascader/plan.cc.o
[ 97%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/ethosu/cascader/plan_generator.cc.o
[ 97%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/ethosu/cascader/propagator.cc.o
[ 98%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/ethosu/cascader/stripe_config.cc.o
[ 98%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/ethosu/cascader/tensor_config.cc.o
[ 98%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/contrib/ethosu/utils.cc.o          EthosU.cmake
[ 98%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/contrib/codegen_c/codegen.cc.o     CODEGENC.cmake
[ 98%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/contrib/example_target_hooks/relay_to_tir.cc.o
[ 98%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/contrib/example_target_hooks/target.cc.o           ExampleTargetHooks.cmake
[100%] Building CXX object CMakeFiles/tvm_objs.dir/src/relay/backend/contrib/example_target_hooks/tir_to_runtime.cc.o
[100%] Building CXX object CMakeFiles/tvm_objs.dir/src/contrib/hybrid/codegen_hybrid.cc.o          HybridDump.cmake




tvm_runtime_objs
[  8%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/builtin_fp16.cc.o
[  8%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/c_runtime_api.cc.o
[  8%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/container.cc.o
[  9%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/cpu_device_api.cc.o
[  9%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/debug.cc.o
[  9%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/dso_library.cc.o
[  9%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/file_utils.cc.o
[ 10%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/library_module.cc.o
[ 11%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/logging.cc.o
[ 11%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/metadata_module.cc.o
[ 11%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/module.cc.o
[ 11%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/ndarray.cc.o
[ 11%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/object.cc.o
[ 11%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/profiling.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/registry.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/source_utils.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/system_library.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/thread_pool.cc.o
[ 12%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/threading_backend.cc.o
[ 13%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/bytecode.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/executable.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/memory_manager.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/vm.cc.o
[ 15%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/workspace_pool.cc.o
[ 16%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_channel.cc.o
[ 16%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_device_api.cc.o
[ 17%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_endpoint.cc.o
[ 17%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_event_impl.cc.o
[ 17%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_local_session.cc.o
[ 17%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_module.cc.o
[ 18%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_pipe_impl.cc.o
[ 18%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_server_env.cc.o
[ 19%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_session.cc.o
[ 19%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/rpc/rpc_socket_impl.cc.o
[ 19%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph_executor/graph_executor.cc.o
[ 19%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph_executor/graph_executor_factory.cc.o
[ 19%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/graph_executor/debug/graph_executor_debug.cc.o
[ 19%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/vm/profiler/vm.cc.o
[ 20%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/cuda/cuda_device_api.cc.o
[ 21%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/cuda/cuda_module.cc.o
[ 21%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/hexagon/hexagon/hexagon_buffer.cc.o
[ 21%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/hexagon/hexagon/hexagon_common.cc.o
[ 21%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/contrib/random/random.cc.o
[ 22%] Building CXX object CMakeFiles/tvm_runtime_objs.dir/src/runtime/contrib/sort/sort.cc.o
```

felix的python内容：

logging.py init_logging(用于初始化Python的日志记录系统，配置同时输出到控制台和日志文件)



注册全局函数
runtime.module.loadfile_stackvm
target.build.stackvm
runtime._datatype_register、 runtime._datatype_get_type_code、runtime._datatype_get_type_name、runtime._datatype_get_type_registered


python里面注册api
runtime:
runtime、node、runtime.profiling

ir
diagnostics、ir、instrument、transform

tir
tir.analysis、tir.schedule、tir.transform、tir.usmp.analysis、tir.usmp.transform、tir.usmp、tir

target
target

te
schedule、tvm.hybrid、te

driver
driver

parser
parser

arith
arith

support
support



onnx.ModelProto是ONNX格式中定义的核心数据结构，用于表示深度学习模型，是ONNX模型的序列化序列
pytorch_lightning.LightningModule是Pytorch Lightning框架中的一个基类，用于封装pytorch模型及其训练逻辑


sym_dag.py文件的内容







































