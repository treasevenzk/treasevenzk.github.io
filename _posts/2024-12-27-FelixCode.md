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


from pathlib import Path
Path处理文件路径

lr_scheduler.MultiStepLR 是Pytorch中用于动态调整学习率的工具，其作用是根据预设的里程牌分阶段降低学习率
lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
optimizer:绑定的优化器对象(如SGD、Adam)     milestones:预设的epoch节点列表(需严格递增)      gamma:衰减系数，每次调整时学习率乘以该系数

@dataclass装饰器：用于自动生成特殊方法如__init__、__repr__、__eq__等
类变量、实例变量：类变量是所有的实例都共享，实例变量定义在dataclass装饰器下，会被自动添加到__init__方法中
@classmethod装饰器：用于定义类方法而不是实例方法，类方法的第一个参数通常命名为cls(而不是self),表示类本身，类方法可以通过类名直接调用，而不需要先创建类的实例


setattr()函数的作用是设置对象的属性值，接受三个参数：对象(要设置属性的对象)、属性名(要设置的属性名称)、属性值(要为属性设置的值)
setattr(object, name, value)

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







felix.extract_tasks