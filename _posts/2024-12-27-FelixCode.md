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
注册的全局函数
***feat_transform.cc***: ~~GetFeaturePack(get_feature_pack---Sketch.fetch_features)~~、LinearExprAsPrimExpr(LinearExpr---RandConfigMaker)
***utils.cc***: ~~ExtractBackbone(extract_backbone---Sketch)~~、~~PrintTrStep(print_state_tr_steps---Sketch)~~、~~GenerateCodeForState(generate_code_for_state---Sketch)~~、**GetLoopBounds(get_loop_bounds)**、ExtractConfigDict(extract_config_dict---add_to_dataset_builder)、~~StateFromConfig(state_from_config---SketchPerfFunc)~~、~~MeasurePerformance(measure_performance---measure_configs_latency_)~~
***sketch_rules.cc***: ~~GenerateAllSymSketches(generate_all_sym_sketches---SymTask)~~
注册的节点： 
***feat_transform***: FeaturePackPyNode、LinearExprNode


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


RELAY_BUILDERS: Conv、DepthwiseConv2d、GroupConv2d、TransposeConv2d、Conv2dTensorCore、Conv2dWinograd、Dense、BatchMatmul、OnePool、TwoOPsPool、ThreeOPsPool、AdaptiveAvgPool、Softmax、Mean、ConstScalar、Elemwise、Broadcast


extract_tasks 经历的过程
sym_tasks.py->utils.py->sym_dag.py


optim.tune 经历的过程
Optimzer->TaskPerfFunc->SketchPerfFunc
SingleTaskOptimizer

_do_one_task_one_round
MLPModelPLWrapper(train_self)->optimizer_round



batch_create_tasks: 
tasks(tasks, task_weights)
task.compute_dag 是访问每个任务的计算图属性

调用全局函数的顺序
SymTask(ffi.generate_all_sym_sketches)
Sketch(ffi.generate_code_for_state、ffi.extract_backbone、ffi.print_state_for_state)
Sketch.fetch_features(ffi.get_feature_pack)--->felix.FeaturePackPy
SketchPerfFunc.make_measure_inputs(ffi.state_from_config)
measure_configs_latency_(ffi.measure_performance)

sym_task
extrac_tasks->extract_tasks_->extract_ansor_tasks->batch_create_tasks->SymbolicDAG.from_ansor->TaskInstance->SymTask(Sketch)->SymTaskAndInstances

utils

sym_dag

optim
MLModelPLWrapper->Timer->TaskPerfFunc(SketchPerfFunc->TorchFeatures-TorchExprRunner)->DatasetBuilder

SingleTaskOptimizer->_do_one_task_one_round->optimize_round(optimize_step -> self.task_f.rounded_sorted_configs->ConfigInfor) -> measure_configs -> measure_config_latency_ -> get_measure_input -> sketch_f.make_measure_inputs

cost_model

features


sym_task ---> utils ---> sym_dag ---> optim ---> cost_model ---> features



def batch_create_tasks(
    tasks: List[utils.AnsorTaskWeight],
    hash_match: bool = True,
    print_tasks: bool = True,
    progress: bool = True,
):
    tasks_ = tqdm(tasks) if progress else tasks
    for i, (task, weight) in enumerate(tasks_):
        concrete_dag = task.compute_dag
        dag_result = SymbolicDAG.from_ansor(concrete_dag, hash_match)
        sym_dag, size_dict = dag_result
        grouped[sym_dag].append(TaskInstance(i, weight, size_dict, task))
    grouped_ = tqdm(grouped.items(), total=len(grouped)) if progress else grouped.items()
    for sym_dag, instances in grouped_:
        indices = [instance.idx for instance in instances]
        size_params, _ = utils.coalesce_dicts([instance.sizes for instance in instances])
        ansor_dag = sym_dag.make_ansor_compute_dag(None)
        task = SymTask(sym_dag, ansor_dag)
        ret_groups.append(SymTaskAndInstances(task, instances))
    return ret_groups

TaskInstance类: idx、weight、sizes、ansor_task
SymTask类: sym_dag、ansor_task、sketches、backbone_to_sketch
Sketch类: parent_task、state_repr、tr_steps、code（生成TIR module）、context、backbone（转换步骤）
Optimizer类: timer、tasks(TaskPerfFunc,weight,idx)、perf_model、n_measure、n_rounds、data_builder、output_file
TaskPerfFunc类: sym_task、sizes、ansor_task、_sketches、perf_model、flops
SketchPerfFunc类: sketch、task_perf_f、features、cost_f
DatasetBuilder类: features、labels、conf_meta
SingleTaskOpitimizer类: task_f、n_seeds、configs、optim、lr_sched、least_lat_history、_dict_conf_hist
ConfigInfo类: config、sketch_f、pred_perf、measure_input、measure_result



ffi.generate_all_sym_sketches(ansor_policy)     SymTask类
ffi.generate_code_for_state(task.ansor_task, sym_state, True)   Sketch类
ffi.extract_backbone
ffi.subst_by_name       SymTask类中的get_flops方法 TaskPerfFunc类
ffi.get_feature_pack    Sketch类中的fetch_features方法 SketchPerfFunc类
ffi.state_from_config   SketchPerfFunc类中的make_measure_inputs方法 measure_configs_latency_方法
ffi.measure_performance measure_configs_latency_方法



felix.GetFeaturePack            ✔
felix.linearExprAsPrimExpr
felix.GenerateAllSymSketches    ✔
felix.ExtractBackBone           ✔
felix.PrintTrStep
felix.GenerateCodeForState      ✔
felix.GetLoopBounds
felix.ExtractConfigDict
felix.StateFromConfig           ✔
felix.MeasurePerformance        ✔


