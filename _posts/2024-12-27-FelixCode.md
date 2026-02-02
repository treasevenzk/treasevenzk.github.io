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


Optimizer类
self.tasks = TaskPerfFunc inst.weight inst.idx


felix经历的过程
extract_tasks

task task_weights
grouped : SymbolicDAC TaskInstance(i, weight, param, task)
task SymTask(sym_dag, ansor_dag)
self.ansor_task ansor_policy = self.make_task_from_dag(sym_ansor_dag)
sketches = ffi.generate_all_sym_sketches

SymTaskAndInstances task instances

SymTask
self.ansor_task self.sketches

SKetch
self.code self.context
self.backbone


SymTaskAndInstances -> SymTask -> Sketch

Optimizer
self.tasks = TaskPerfFunc(SymTask, TaskInstance.sizes, MLPModelPLWrapper)





FeaturePack
variables: 多阶段变量存储容器
linear_cons: 线性约束管理
split_groups: 循环分割信息
var_factors: 质数分解映射
no_factoring: 控制标志

RunRollingSimplify(): 滚动简化优化表达式， 如原始: x = a * b , y = x * c 简化后: y = a * b * c
RunExDecompose(): 质数幂次分解  任意正整数n可分解为: n = 2^a * 3^ b * 5^c * 7^d * ... Felix将任意sp
RunDiffTransform(): 可微分变换
RunSizeSubstitution(): 具体尺寸替换
IntoPythonFeaturePack(): 转换为Python对象

vdefs: 所有变量定义
features: 提取的特征表达式
constraints: 约束条件
ei_vars: 形状变量(shape variables), 如E0、E1(表示输入tensor的维度)
vi_vars: 中间变量(Intermediate Variables)
exp_decomp: 指数分解后的表达式
vi_and_features: vi_vars和features的合并
diff_approx: 可微近似表达式

TIR Stmt + VarContext
        ↓
StmtSimplifier(简化语句)
        ↓
GetPerStoreFeatureExpr(提取特征)
GetConstraints(提取约束)
        ↓
构造 FeaturePack(vdefs, features, constriants, split_groups)
        ↓
RunRollingSimplify(简化表达式) 拆分: vdefs → ei_vars + vi_vars
                             合并: vi_vars + features → vi_and_features
        ↓
RunExpDecompose(指数分解) 生成exp_decomp, var_factors
        ↓
RunDiffTransform(可微分转换) 移除: floor
                           应用: DiffableApprox
                           生成: diff_approx, linear_cons
        ↓
保存/加载JSON
        ↓
RunSizeSubstitution(尺寸替换) 替换: ei_vars
                            计算: split_groups_factos
                            合并所有变量到: vdefs
                            简化: linear_cons
        ↓
IntoPythonFeaturePack(转换为Python对象)
        ↓
FeaturePackPy(expressions, free_vars, linear_cons, var_factors)



self.ansor_task, ansor_policy = self.make_task_from_dag(sym_ansor_dag)
sketches = ffi.generate_all_sym_sketches(ansor_policy)
self.sketches = [Sketch(self, sketch) for sketch in sketches]
self.code self.context = ffi.generate_code_for_state(task.ansor_task, sym_state, True)
self.backbone = ffi.extract_backbone(self.tr_steps)
self.features = sketch.fetch_features(task_perf_f.sizes)
features = ffi.get_feature_pack(self.code, self.context, HW_PARAMS, sizes, cache_line_sizes, max_n_buf, kprime_factorize, path.with_suffic(").as_posix())
ansor.MeasureInput(task, ffi.state_from_config(task, self.sketch.tr_steps, c))

ffi.measure_performance(measurer, [c.get_measure_input() for c in configs])


["CHW", 2, "local"], 
["SP", 2, 0, 512, [1, 4, 16, 1], 1], 
["SP", 2, 5, 4096, [1, 16, 2, 4], 1], 
["SP", 2, 10, 1024, [2, 2], 1], 
["RE", 2, [0, 5, 1, 6, 2, 7, 10, 11, 3, 8, 12, 4, 9]], SSSRRSRS

["FSP", 3, 0, 1, 3], 
["FSP", 3, 4, 2, 3], 
["RE", 3, [0, 4, 1, 5, 2, 6, 3, 7]], 
["CA", 2, 3, 5], 

["CHR", 1, "shared", [2]], 
["CA", 2, 3, 6], 
["CHR", 0, "shared", [3]], 
["CA", 1, 4, 6], 

["FU", 5, [0, 1]], 
["AN", 5, 0, 5], 
["FU", 5, [1, 2]], 
["AN", 5, 1, 4], 
["FU", 5, [2, 3]], 
["AN", 5, 2, 6], 

["FU", 3, [0, 1]], 
["SP", 3, 0, 32, [2], 1], 
["AN", 3, 1, 2], 
["FFSP", 3, 0, [2, 1], 1, 1], 
["AN", 3, 1, 6], 

["FU", 1, [0, 1]], 
["SP", 1, 0, 64, [2], 1], 
["AN", 1, 1, 2], 
["FFSP", 1, 0, [2, 1], 1, 1], 
["AN", 1, 1, 6], 

["PR", 4, 0, "auto_unroll_max_step$1024"]sour


AN      AnnotationStep:         record_prefix_str、stage_id、iter_id、
FU      FuseStep:               record_prefix_str、stage_id、fused_ids
PR      PragmaStep:             record_prefix_str、stage_id、iter_id、          pragma_type
RE      ReorderStep:            record_prefix_str、stage_id、after_ids
SP      SplitStep:              record_prefix_str、stage_id、iter_id、          extent、        lengths、       inner_to_outer
FSP     FolllowSplitStep:       record_prefix_str、stage_id、iter_id、          src_step_id、   n_split

FFSP    FollowFusedSplitStep:   record_prefix_str、stage_id、iter_id、          src_step_id、   level、         factor_or_nparts
SA      StorageAlignStep:       record_prefix_str、stage_id、iter_id、          factor、        offset
CA      ComputeAtStep:          record_prefix_str、stage_id、target_stage_id、  target_iter_id
CI      ComputeInlineStep:      record_prefix_str、stage_id
CR      ComputeRootStep:        record_prefix_str、stage_id
CHR     CacheReadStep:          record_prefix_str、stage_id、scope_name、       reader_stage_ids
CHW     CacheWriteStep:         record_prefix_str、stage_id、scope_name
RF      RfactorStep             record_prefix_str、stage_id、iter_id            factor_iter_id





["CHW", 2, "local"], 
["SP", 2, 0, 512, [1, 4, 16, 1], 1], 
["SP", 2, 5, 4096, [1, 16, 2, 4], 1], 
["SP", 2, 10, 1024, [4, 2], 1], 
["RE", 2, [0, 5, 1, 6, 2, 7, 10, 11, 3, 8, 12, 4, 9]], 

["FSP", 3, 0, 1, 3], 
["FSP", 3, 4, 2, 3], 
["RE", 3, [0, 4, 1, 5, 2, 6, 3, 7]], 
["CA", 2, 3, 5], 
["CHR", 1, "shared", [2]], 
["CA", 2, 3, 6], 
["CHR", 0, "shared", [3]], 
["CA", 1, 4, 6],

["FU", 5, [0, 1]], 
["AN", 5, 0, 5], 
["FU", 5, [1, 2]], 
["AN", 5, 1, 4], 
["FU", 5, [2, 3]], 
["AN", 5, 2, 6],

["FU", 3, [0, 1]], 
["SP", 3, 0, 64, [2], 1], 
["AN", 3, 1, 2], 
["FFSP", 3, 0, [2, 1], 1, 1], 
["AN", 3, 1, 6], 

["FU", 1, [0, 1]], 
["SP", 1, 0, 128, [2], 1], 
["AN", 1, 1, 2], 
["FFSP", 1, 0, [2, 1], 1, 1], 
["AN", 1, 1, 6], 

["PR", 4, 0, "auto_unroll_max_step$1024"]

('resnet_50', [(1, 3, 224, 224)])
[09:44:15] /home/zk/Pruner/src/runtime/threading_backend.cc:217: warning: more than two frequencies detected!
========== Task 0  (workload key: ["9847f8cc0b305137f49f2c5c0c8ab25d", 1, 2048, 1000, 2048, 1000, 1, 1000]...) ==========
placeholder = PLACEHOLDER [1, 2048]
placeholder = PLACEHOLDER [1000, 2048]
T_dense(i, j) += (placeholder[i, k]*placeholder[j, k])
placeholder = PLACEHOLDER [1000]
T_add(ax0, ax1) = (T_dense[ax0, ax1] + placeholder[ax1])

6.27229e-05


========== Task 1  (workload key: ["69115f188984ae34ede37c3b8ca40b43", 1, 7, 7, 2048, 1, 1, 1, 2048]...) ==========
placeholder = PLACEHOLDER [1, 7, 7, 2048]
tensor(ax0, ax1, ax2, ax3) += placeholder[ax0, ((ax1*7) + rv0), ((ax2*7) + rv1), ax3]
tensor(ax0, ax1, ax2, ax3) = (tensor[ax0, ax1, ax2, ax3]/(float32((select((bool)1, ((ax1 + 1)*7), (((ax1 + 1)*7) + 1)) - (ax1*7)))*float32((select((bool)1, ((ax2 + 1)*7), (((ax2 + 1)*7) + 1)) - (ax2*7)))))

3.62703e-06

========== Task 2  (workload key: ["022e9ad925fa9d131944586979dafc0f", 1, 7, 7, 512, 1, 1, 512, 2048, 1, 1, 1, 2048, 1, 7, 7, 2048, 1, 7, 7, 2048]...) 3==========
placeholder = PLACEHOLDER [1, 7, 7, 512]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 512, 2048]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 2048]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
placeholder = PLACEHOLDER [1, 7, 7, 2048]
T_add(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

1.99834e-05   80
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 7, [1, 1, 7, 1], 1], ["SP", 3, 10, 7, [1, 7, 1, 1], 1], ["SP", 3, 15, 2048, [1, 8, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 512, [8, 16], 1], 

========== Task 7  (workload key: ["022e9ad925fa9d131944586979dafc0f", 1, 14, 14, 256, 1, 1, 256, 1024, 1, 1, 1, 1024, 1, 14, 14, 1024, 1, 14, 14, 1024]...) 6==========
placeholder = PLACEHOLDER [1, 14, 14, 256]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 1024]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 1024]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
placeholder = PLACEHOLDER [1, 14, 14, 1024]
T_add(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

1.52316e-05 120
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 14, [1, 1, 1, 2], 1], ["SP", 3, 10, 14, [1, 7, 1, 2], 1], ["SP", 3, 15, 1024, [1, 16, 4, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 256, [32, 2], 1],


========== Task 12  (workload key: ["022e9ad925fa9d131944586979dafc0f", 1, 28, 28, 128, 1, 1, 128, 512, 1, 1, 1, 512, 1, 28, 28, 512, 1, 28, 28, 512]...) 4==========
placeholder = PLACEHOLDER [1, 28, 28, 128]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 128, 512]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
placeholder = PLACEHOLDER [1, 28, 28, 512]
T_add(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

1.42663e-05 80
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 28, [1, 2, 7, 1], 1], ["SP", 3, 10, 28, [1, 4, 1, 1], 1], ["SP", 3, 15, 512, [4, 32, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 128, [8, 4], 1]


========== Task 17  (workload key: ["022e9ad925fa9d131944586979dafc0f", 1, 56, 56, 64, 1, 1, 64, 256, 1, 1, 1, 256, 1, 56, 56, 256, 1, 56, 56, 256]...) 3==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 64, 256]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
placeholder = PLACEHOLDER [1, 56, 56, 256]
T_add(ax0, ax1, ax2, ax3) = (T_add[ax0, ax1, ax2, ax3] + placeholder[ax0, ax1, ax2, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

1.69547e-05 60
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 56, [1, 1, 2, 2], 1], ["SP", 3, 10, 56, [1, 14, 1, 2], 1], ["SP", 3, 15, 256, [2, 16, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 64, [16, 4], 1]

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1.99834e-05   80
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 7, [1, 1, 7, 1], 1], ["SP", 3, 10, 7, [1, 7, 1, 1], 1], ["SP", 3, 15, 2048, [1, 8, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 512, [8, 16], 1], 

1.42663e-05 80
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 28, [1, 2, 7, 1], 1], ["SP", 3, 10, 28, [1, 4, 1, 1], 1], ["SP", 3, 15, 512, [4, 32, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 128, [8, 4], 1]

"CI" "CI" 
"SP" "SP" "SP" "SP" "SP" "SP" "SP" "RE" 
"FSP" "FSP" "FSP" "FSP" "RE" 
"CA" 
"CHR" "CA" "CHR" "CA" 
"CI" 
"FU" "AN" "FU" "AN" "FU" "AN" 
"FU" "SP" "AN" "FFSP" "AN" 
"FU" "SP" "AN" "FFSP" "AN" 
"PR" "auto_unroll_max_step$64"


========== Task 3  (workload key: ["3a69f9fbc63760d99e36b4c17b3bfc57", 1, 7, 7, 512, 4, 4, 512, 512, 1, 1, 1, 512, 1, 7, 7, 512]...) 2==========
placeholder = PLACEHOLDER [1, 7, 7, 512]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 8)) && (i2 >= 1)) && (i2 < 8)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 16), ((floormod(floordiv(p, 4), 4)*2) + eps), ((floormod(p, 4)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 512, 512]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*4)*4) + (floordiv(h, 2)*4)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

5.46359e-05 140


========== Task 4  (workload key: ["6b7583cf23c7c37d3212cad9d06e58c1", 1, 7, 7, 2048, 1, 1, 2048, 512, 1, 1, 1, 512, 1, 7, 7, 512]...) 2==========
placeholder = PLACEHOLDER [1, 7, 7, 2048]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 2048, 512]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

4.04766e-05 100
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 7, [1, 7, 1, 1], 1], ["SP", 3, 10, 7, [1, 1, 1, 1], 1], ["SP", 3, 15, 512, [1, 8, 1, 4], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 2048, [16, 8], 1], 

========== Task 6  (workload key: ["6b7583cf23c7c37d3212cad9d06e58c1", 1, 14, 14, 1024, 1, 1, 1024, 512, 1, 1, 1, 512, 1, 14, 14, 512]...) ==========
placeholder = PLACEHOLDER [1, 14, 14, 1024]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 1024, 512]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

3.13058e-05 50
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 14, [1, 1, 7, 1], 1], ["SP", 3, 10, 14, [1, 7, 1, 1], 1], ["SP", 3, 15, 512, [1, 16, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 1024, [2, 64], 1], 

========== Task 9  (workload key: ["6b7583cf23c7c37d3212cad9d06e58c1", 1, 14, 14, 1024, 1, 1, 1024, 256, 1, 1, 1, 256, 1, 14, 14, 256]...) 5==========
placeholder = PLACEHOLDER [1, 14, 14, 1024]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 1024, 256]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

2.1559e-05 140
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 14, [1, 7, 2, 1], 1], ["SP", 3, 10, 14, [1, 1, 2, 1], 1], ["SP", 3, 15, 256, [1, 16, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 1024, [2, 64], 1], 

========== Task 11  (workload key: ["6b7583cf23c7c37d3212cad9d06e58c1", 1, 28, 28, 512, 1, 1, 512, 256, 1, 1, 1, 256, 1, 28, 28, 256]...) ==========
placeholder = PLACEHOLDER [1, 28, 28, 512]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 512, 256]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

2.74661e-05 50
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 28, [1, 7, 2, 2], 1], ["SP", 3, 10, 28, [1, 1, 2, 1], 1], ["SP", 3, 15, 256, [2, 32, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 512, [32, 2], 1],



========== Task 14  (workload key: ["6b7583cf23c7c37d3212cad9d06e58c1", 1, 28, 28, 512, 1, 1, 512, 128, 1, 1, 1, 128, 1, 28, 28, 128]...) 3==========
placeholder = PLACEHOLDER [1, 28, 28, 512]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 512, 128]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

1.99953e-05 80
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 28, [1, 4, 1, 1], 1], ["SP", 3, 10, 28, [2, 1, 2, 1], 1], ["SP", 3, 15, 128, [1, 32, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 512, [8, 8], 1],

========== Task 16  (workload key: ["6b7583cf23c7c37d3212cad9d06e58c1", 1, 56, 56, 256, 1, 1, 256, 128, 1, 1, 1, 128, 1, 56, 56, 128]...) ==========
placeholder = PLACEHOLDER [1, 56, 56, 256]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 128]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

2.50949e-05 100
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 56, [1, 1, 1, 1], 1], ["SP", 3, 10, 56, [1, 2, 14, 1], 1], ["SP", 3, 15, 128, [4, 32, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 256, [4, 4], 1], 


========== Task 19  (workload key: ["6b7583cf23c7c37d3212cad9d06e58c1", 1, 56, 56, 256, 1, 1, 256, 64, 1, 1, 1, 64, 1, 56, 56, 64]...) 2==========
placeholder = PLACEHOLDER [1, 56, 56, 256]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 64]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

1.50793e-05 50
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 56, [1, 1, 1, 1], 1], ["SP", 3, 10, 56, [1, 7, 4, 1], 1], ["SP", 3, 15, 64, [2, 16, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 256, [16, 4], 1], 


========== Task 20  (workload key: ["6b7583cf23c7c37d3212cad9d06e58c1", 1, 56, 56, 64, 1, 1, 64, 64, 1, 1, 1, 64, 1, 56, 56, 64]...) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 64, 64]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

5.01663e-06 10
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 56, [2, 2, 1, 1], 1], ["SP", 3, 10, 56, [1, 4, 1, 2], 1], ["SP", 3, 15, 64, [1, 16, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 64, [32, 2], 1], 


========== Task 5  (workload key: ["12b88bedece6984af589a28b43e0f3c4", 1, 14, 14, 512, 3, 3, 512, 512, 1, 1, 1, 512, 1, 7, 7, 512]...) ==========
placeholder = PLACEHOLDER [1, 14, 14, 512]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 15)) && (i2 >= 1)) && (i2 < 15)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 512, 512]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

7.9156e-05 100
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 7, [1, 7, 1, 1], 1], ["SP", 3, 10, 7, [1, 1, 1, 7], 1], ["SP", 3, 15, 512, [1, 16, 1, 1], 1], ["SP", 3, 20, 3, [1, 3], 1], ["SP", 3, 23, 3, [1, 3], 1], ["SP", 3, 26, 512, [8, 2], 1],

========== Task 10  (workload key: ["12b88bedece6984af589a28b43e0f3c4", 1, 28, 28, 256, 3, 3, 256, 256, 1, 1, 1, 256, 1, 14, 14, 256]...) ==========
placeholder = PLACEHOLDER [1, 28, 28, 256]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 29)) && (i2 >= 1)) && (i2 < 29)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 256, 256]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

4.67301e-05 70
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 14, [1, 7, 2, 1], 1], ["SP", 3, 10, 14, [1, 1, 2, 1], 1], ["SP", 3, 15, 256, [1, 16, 2, 1], 1], ["SP", 3, 20, 3, [1, 3], 1], ["SP", 3, 23, 3, [1, 1], 1], ["SP", 3, 26, 256, [2, 32], 1],

========== Task 15  (workload key: ["12b88bedece6984af589a28b43e0f3c4", 1, 56, 56, 128, 3, 3, 128, 128, 1, 1, 1, 128, 1, 28, 28, 128]...) ==========
placeholder = PLACEHOLDER [1, 56, 56, 128]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 57)) && (i2 >= 1)) && (i2 < 57)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
placeholder = PLACEHOLDER [3, 3, 128, 128]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

4.1521e-05 50
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 28, [1, 1, 1, 1], 1], ["SP", 3, 10, 28, [1, 14, 2, 1], 1], ["SP", 3, 15, 128, [1, 16, 2, 1], 1], ["SP", 3, 20, 3, [1, 3], 1], ["SP", 3, 23, 3, [3, 1], 1], ["SP", 3, 26, 128, [8, 2], 1],


========== Task 22  (workload key: ["12b88bedece6984af589a28b43e0f3c4", 1, 224, 224, 3, 7, 7, 3, 64, 1, 1, 1, 64, 1, 112, 112, 64]...) ==========
placeholder = PLACEHOLDER [1, 224, 224, 3]
PaddedInput(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 3) && (i1 < 227)) && (i2 >= 3)) && (i2 < 227)), placeholder[i0, (i1 - 3), (i2 - 3), i3], 0f)
placeholder = PLACEHOLDER [7, 7, 3, 64]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

2.34164e-05 50
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 112, [1, 7, 2, 1], 1], ["SP", 3, 10, 112, [1, 1, 2, 2], 1], ["SP", 3, 15, 64, [2, 32, 1, 1], 1], ["SP", 3, 20, 7, [7, 1], 1], ["SP", 3, 23, 7, [1, 7], 1], ["SP", 3, 26, 3, [1, 1], 1],








========== Task 8  (workload key: ["b8b52b9be9df6102466a22a014c44c1f", 1, 14, 14, 256, 4, 4, 256, 256, 1, 1, 1, 256, 1, 14, 14, 256]...) 5==========
placeholder = PLACEHOLDER [1, 14, 14, 256]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 15)) && (i2 >= 1)) && (i2 < 15)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 49), ((floormod(floordiv(p, 7), 7)*2) + eps), ((floormod(p, 7)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 256, 256]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*7)*7) + (floordiv(h, 2)*7)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

2.9158e-05 180

========== Task 13  (workload key: ["e4cdf917b876dbdd64488c3818d9c141", 1, 28, 28, 128, 4, 4, 128, 128, 1, 1, 1, 128, 1, 28, 28, 128]...) 3==========
placeholder = PLACEHOLDER [1, 28, 28, 128]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 29)) && (i2 >= 1)) && (i2 < 29)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 196), ((floormod(floordiv(p, 14), 14)*2) + eps), ((floormod(p, 14)*2) + nu), ci]
B(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 4) == 2)),  ..(OMITTED).. ormod(i, 4) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [4, 4, 128, 128]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 4) == 3) && (floormod(j, 2) == 1)), 1f, select(((floormod(i, 4) == 3) && (floormod(j, 2) == 0)),  ..(OMITTED).. ct(((floormod(i, 4) == 0) && (floormod(j, 2) == 1)), 0f, select(((floormod(i, 4) == 0) && (floormod(j, 2) == 0)), 1f, 0f))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 2), floormod(w, 2), ((((n*14)*14) + (floordiv(h, 2)*14)) + floordiv(w, 2)), co]
placeholder = PLACEHOLDER [1, 1, 1, 128]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

3.42814e-05 100

========== Task 18  (workload key: ["b818b53148cd450f86569dfc3e04cb8a", 1, 56, 56, 64, 6, 6, 64, 64, 1, 1, 1, 64, 1, 56, 56, 64]...) 3==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
data_pad(i0, i1, i2, i3) = tir.if_then_else(((((i1 >= 1) && (i1 < 57)) && (i2 >= 1)) && (i2 < 57)), placeholder[i0, (i1 - 1), (i2 - 1), i3], 0f)
input_tile(eps, nu, p, ci) = data_pad[floordiv(p, 196), ((floormod(floordiv(p, 14), 14)*4) + eps), ((floormod(p, 14)*4) + nu), ci]
B(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 6) == 5)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 6) == 4)),  ..(OMITTED)..  (floormod(j, 6) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 6) == 0)), 1f, 0f))))))))))))))))))))))))))))))))))))
data_pack(eps, nu, p, ci) += ((input_tile[r_a, r_b, p, ci]*B[r_a, eps])*B[r_b, nu])
placeholder = PLACEHOLDER [6, 6, 64, 64]
bgemm(eps, nu, p, co) += (data_pack[eps, nu, p, ci]*placeholder[eps, nu, co, ci])
A(i, j) = select(((floormod(i, 6) == 5) && (floormod(j, 4) == 3)), 1f, select(((floormod(i, 6) == 5) && (floormod(j, 4) == 2)),  ..(OMITTED)..  6) == 0) && (floormod(j, 4) == 1)), 0f, select(((floormod(i, 6) == 0) && (floormod(j, 4) == 0)), 1f, 0f))))))))))))))))))))))))
inverse(vh, vw, p, co) += ((bgemm[r_a, r_b, p, co]*A[r_a, vh])*A[r_b, vw])
conv2d_winograd(n, h, w, co) = inverse[floormod(h, 4), floormod(w, 4), ((((n*14)*14) + (floordiv(h, 4)*14)) + floordiv(w, 4)), co]
placeholder = PLACEHOLDER [1, 1, 1, 64]
T_add(ax0, ax1, ax2, ax3) = (conv2d_winograd[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])
T_relu(ax0, ax1, ax2, ax3) = max(T_add[ax0, ax1, ax2, ax3], 0f)

2.29611e-05 100

========== Task 21  (workload key: ["23b8a36aee55cd70eeb0706ebd25dcf3", 1, 112, 112, 64, 1, 56, 56, 64]...) ==========
placeholder = PLACEHOLDER [1, 112, 112, 64]
pad_temp(ax0, ax1, ax2, ax3) = tir.if_then_else(((((ax1 >= 1) && (ax1 < 113)) && (ax2 >= 1)) && (ax2 < 113)), placeholder[ax0, (ax1 - 1), (ax2 - 1), ax3], -3.40282e+38f)
tensor(ax0, ax1, ax2, ax3) max= pad_temp[ax0, ((ax1*2) + dh), ((ax2*2) + dw), ax3]

5.75706e-06 10

========== Task 23  (workload key: ["9f4c6b76f51d20e5c3bfb1817edd446e", 1, 56, 56, 64, 1, 1, 64, 256, 1, 1, 1, 256, 1, 56, 56, 256]...) ==========
placeholder = PLACEHOLDER [1, 56, 56, 64]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 64, 256]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, (yy + ry), (xx + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 256]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])

1.41949e-05 20
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 56, [1, 1, 2, 1], 1], ["SP", 3, 10, 56, [1, 1, 1, 28], 1], ["SP", 3, 15, 256, [1, 128, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 64, [4, 4], 1], 


========== Task 24  (workload key: ["a12fbee7636d485f5404304276246b54", 1, 56, 56, 256, 1, 1, 256, 512, 1, 1, 1, 512, 1, 28, 28, 512]...) ==========
placeholder = PLACEHOLDER [1, 56, 56, 256]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 256, 512]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 512]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])

2.84199e-05 50
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 28, [1, 4, 1, 1], 1], ["SP", 3, 10, 28, [1, 1, 7, 1], 1], ["SP", 3, 15, 512, [2, 32, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 256, [2, 8], 1]


========== Task 25  (workload key: ["a12fbee7636d485f5404304276246b54", 1, 28, 28, 512, 1, 1, 512, 1024, 1, 1, 1, 1024, 1, 14, 14, 1024]...) ==========
placeholder = PLACEHOLDER [1, 28, 28, 512]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 512, 1024]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 1024]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])

3.62962e-05 50
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 14, [1, 1, 7, 2], 1], ["SP", 3, 10, 14, [1, 1, 1, 1], 1], ["SP", 3, 15, 1024, [1, 128, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 512, [16, 4], 1], 

========== Task 26  (workload key: ["a12fbee7636d485f5404304276246b54", 1, 14, 14, 1024, 1, 1, 1024, 2048, 1, 1, 1, 2048, 1, 7, 7, 2048]...) ==========
placeholder = PLACEHOLDER [1, 14, 14, 1024]
PaddedInput(i0, i1, i2, i3) = placeholder[i0, i1, i2, i3]
placeholder = PLACEHOLDER [1, 1, 1024, 2048]
Conv2dOutput(nn, yy, xx, ff) += (PaddedInput[nn, ((yy*2) + ry), ((xx*2) + rx), rc]*placeholder[ry, rx, rc, ff])
placeholder = PLACEHOLDER [1, 1, 1, 2048]
T_add(ax0, ax1, ax2, ax3) = (Conv2dOutput[ax0, ax1, ax2, ax3] + placeholder[ax0, 0, 0, ax3])

5.49496e-05 70
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 7, [1, 1, 1, 1], 1], ["SP", 3, 10, 7, [1, 1, 1, 7], 1], ["SP", 3, 15, 2048, [1, 64, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 1024, [4, 16], 1],



2、7、12、17

1.99834e-05   80
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 7, [1, 1, 7, 1], 1], ["SP", 3, 10, 7, [1, 7, 1, 1], 1], ["SP", 3, 15, 2048, [1, 8, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 512, [8, 16], 1], 

1.52316e-05 120
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 14, [1, 1, 1, 2], 1], ["SP", 3, 10, 14, [1, 7, 1, 2], 1], ["SP", 3, 15, 1024, [1, 16, 4, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 256, [32, 2], 1],

1.42663e-05 80
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 28, [1, 2, 7, 1], 1], ["SP", 3, 10, 28, [1, 4, 1, 1], 1], ["SP", 3, 15, 512, [4, 32, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 128, [8, 4], 1]

1.69547e-05 60
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 56, [1, 1, 2, 2], 1], ["SP", 3, 10, 56, [1, 14, 1, 2], 1], ["SP", 3, 15, 256, [2, 16, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 64, [16, 4], 1]

4、6、9、11、14、16、19、20、5、10、15、22

4.04766e-05 100         4
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 7, [1, 7, 1, 1], 1], ["SP", 3, 10, 7, [1, 1, 1, 1], 1], ["SP", 3, 15, 512, [1, 8, 1, 4], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 2048, [16, 8], 1], 

3.13058e-05 50          6
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 14, [1, 1, 7, 1], 1], ["SP", 3, 10, 14, [1, 7, 1, 1], 1], ["SP", 3, 15, 512, [1, 16, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 1024, [2, 64], 1], 


2.1559e-05 140          9
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 14, [1, 7, 2, 1], 1], ["SP", 3, 10, 14, [1, 1, 2, 1], 1], ["SP", 3, 15, 256, [1, 16, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 1024, [2, 64], 1], 

2.74661e-05 50          11
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 28, [1, 7, 2, 2], 1], ["SP", 3, 10, 28, [1, 1, 2, 1], 1], ["SP", 3, 15, 256, [2, 32, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 512, [32, 2], 1],

1.99953e-05 80          14
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 28, [1, 4, 1, 1], 1], ["SP", 3, 10, 28, [2, 1, 2, 1], 1], ["SP", 3, 15, 128, [1, 32, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 512, [8, 8], 1],

2.50949e-05 100         16
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 56, [1, 1, 1, 1], 1], ["SP", 3, 10, 56, [1, 2, 14, 1], 1], ["SP", 3, 15, 128, [4, 32, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 256, [4, 4], 1], 

1.50793e-05 50          19
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 56, [1, 1, 1, 1], 1], ["SP", 3, 10, 56, [1, 7, 4, 1], 1], ["SP", 3, 15, 64, [2, 16, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 256, [16, 4], 1], 

5.01663e-06 10          20
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 56, [2, 2, 1, 1], 1], ["SP", 3, 10, 56, [1, 4, 1, 2], 1], ["SP", 3, 15, 64, [1, 16, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 64, [32, 2], 1], 

7.9156e-05 100          5
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 7, [1, 7, 1, 1], 1], ["SP", 3, 10, 7, [1, 1, 1, 7], 1], ["SP", 3, 15, 512, [1, 16, 1, 1], 1], ["SP", 3, 20, 3, [1, 3], 1], ["SP", 3, 23, 3, [1, 3], 1], ["SP", 3, 26, 512, [8, 2], 1],

4.67301e-05 70          10
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 14, [1, 7, 2, 1], 1], ["SP", 3, 10, 14, [1, 1, 2, 1], 1], ["SP", 3, 15, 256, [1, 16, 2, 1], 1], ["SP", 3, 20, 3, [1, 3], 1], ["SP", 3, 23, 3, [1, 1], 1], ["SP", 3, 26, 256, [2, 32], 1],

4.1521e-05 50           15
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 28, [1, 1, 1, 1], 1], ["SP", 3, 10, 28, [1, 14, 2, 1], 1], ["SP", 3, 15, 128, [1, 16, 2, 1], 1], ["SP", 3, 20, 3, [1, 3], 1], ["SP", 3, 23, 3, [3, 1], 1], ["SP", 3, 26, 128, [8, 2], 1],

2.34164e-05 50          22
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 112, [1, 7, 2, 1], 1], ["SP", 3, 10, 112, [1, 1, 2, 2], 1], ["SP", 3, 15, 64, [2, 32, 1, 1], 1], ["SP", 3, 20, 7, [7, 1], 1], ["SP", 3, 23, 7, [1, 7], 1], ["SP", 3, 26, 3, [1, 1], 1],


23、24、25、26

1.41949e-05 20
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 56, [1, 1, 2, 1], 1], ["SP", 3, 10, 56, [1, 1, 1, 28], 1], ["SP", 3, 15, 256, [1, 128, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 64, [4, 4], 1], 

2.84199e-05 50
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 28, [1, 4, 1, 1], 1], ["SP", 3, 10, 28, [1, 1, 7, 1], 1], ["SP", 3, 15, 512, [2, 32, 2, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 256, [2, 8], 1]

3.62962e-05 50
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 14, [1, 1, 7, 2], 1], ["SP", 3, 10, 14, [1, 1, 1, 1], 1], ["SP", 3, 15, 1024, [1, 128, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 512, [16, 4], 1], 

5.49496e-05 70
["SP", 3, 0, 1, [1, 1, 1, 1], 1], ["SP", 3, 5, 7, [1, 1, 1, 1], 1], ["SP", 3, 10, 7, [1, 1, 1, 7], 1], ["SP", 3, 15, 2048, [1, 64, 1, 1], 1], ["SP", 3, 20, 1, [1, 1], 1], ["SP", 3, 23, 1, [1, 1], 1], ["SP", 3, 26, 1024, [4, 16], 1],