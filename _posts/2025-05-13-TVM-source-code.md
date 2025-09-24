---
layout:     post
title:      TVM
subtitle:   TVM source code
date:       2025-05-13
author:     Treaseven
header-img: img/bg2.jpg
catalog: true
tags:
    - code analysis
---

关键属性<br>
**runtime**
```
c_runtime_api.h:
TVM_DLL：标记函数/类需要对库的使用者可见
TVMArgTypeCode、TVMArrayHandle、TVMValue、TVMByteArray、TVMModuleHandle、TVMFunctionHandle、TVMRetValueHandle、TVMStreamHandle、TVMObjectHandle
```

```
container.h
ObjectHash、ObjectEqual、InplaceArrayBase、IterAdapter、ReverseIterAdapter、ArrayNode、Array、ADTObj、AD、StringObj、String、NullOptType、Optional、ClosureObj、CLosure
```

```
data_type.h
DataType
```

```
memory.h
ObjAllocatorBase、SimpleObjAllocator
```

```
module.h
Module、ModuleNode
```

```
ndarray.h
NDArray
```

```
object.h
TypeIndex、Object、ObjectPtr、ObjectRef、ObjectPtrHash、ObjectPtrEqual
TVM_DECLARE_BASE_OBJECT_INFO(TypeName, ParentType)
TVM_DECLARE_FINAL_OBJECT_INFO(TypeName, ParentType)
TVM_OBJECT_REG_VAR_DEF
TVM_REGISTER_OBJECT_TYPE(TypeName)
TVM_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(TypeName)
TVM_DEFINE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)
TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName) 
TVM_DEFINE_MUTALBLE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName)
TVM_DEFINE_OBJECT_REF_COW_METHOD(TypeName, ParentType, ObjectName) 
```

```
packed_func.h
PackedFunc、TypedPackedFunc、TVMArgs、ObjectTypeChecker、TVMPODValue_、TVMMovableArgValue_、TVMRetValue、PackedFuncValueConverter、TVMArgsSetter
```

```
registry.h
Registry
TVM_FUNC_REG_VAR_DEF
TVm_REGISTER_GLOBAL(OpName)
TVM_STRINGIZE_DETAL
TVM_STRINGIZE
TVM_DESCRIBE
TVM_ADD_FILEINE
```

```
threading_backend.h
ThreadGroup
```

**te**
```
operation.h
TensorDom、OperationNode、PlaceholderOpNode、PlaceholderOp、BaseComputeOpNode、ComputeOpNode、ComputeOp、TensorComputeOpNode、TensorComputeOp、ScanOpNode、ScanOp、ExternOpNode、ExternOp、HybridOpNode、HybridOp
```

```
schedule.h
AttachType、Stage、Schedule、IterVarRelation、IterVarAttr、StageNode、ScheduleNode、IterVarAttrNode、IterVarRelationNode、SplitNode、Split、FuseNode、Fuse、RebaseNode、Rebase、SingletonNode、Singleton、SpecializedConditionNode、SpecializedCondition
```

```
tensor_intrin.h
TensorIntrinNode、TensorIntrin、TensorIntrinCallNode、TensorIntrinCall
```

```
tensor.h
Operation、TensorNode、Tensor
```

**ir**
```
adt.h
ConstructorNode、Constructor、TypeDataNode、TypeData
```

```
attrs.h
AttrError、AttrFieldInfoNode、AttrFieldInfo、BaseAttrsNode、Attrs、DictAttrsNode、DictAttrs、AttrNopEntry、AttrNormalVisitor、AttrsEqualVisitor、AttrsHashVisitor、AttrInitEntry、AttrInitVisitor、AttrInitVisitor、TypeName、AttrDocEntry、AttrDocVisitor、AttrsExistsVisitor、AttrTriggerNonDefaultEntry、AttrNonDefaultVisitor、AttrsNode
```

```
diagnostic.h
DiagnosticLevel、DiagnosticBuilder、DiagnosticNode、Diagnostic、DiagnosticRenderNode、DiagnosticRender、DiagnosticContextNode、DiagnosticContext
```

```
env_func.h
EnvFuncNode、EnvFunc、TypedEnvFunc
```

```
error.h
ErrorBuilder、CompileError、ErrorReporter
```

```
expr.h
BaseExprNode、BaseExpr、PrimExprNode、PrimExpr、RelayExprNode、RelayExpr、GlobalVarNode、GlobalVar、IntImmNode、IntImm、FloatImmNode、FloatImm、Bool、Integer、RangeNode、RangePackedFuncValueConverter
```

```
function.h
CallingConv、BaseFuncNode、BaseFunc、
```

```
module.h
IRModuleNode、IRModule
```

```
op.h
OpNode、Op、OpRegEntry、OpAttrMap、
```

```
span.h
SourceNameNode、SourceName、SpanNode、Span
```

```
tensor_type.h
BaseTensorTypeNode、BaseTensorType、TensorTypeNode、TensorType、GenericTensorType、GenericDataType、GenericShape
```

```
transform.h
PassContextNode、PassContext、PassInfoNode、PassInfo、Pass、SequentialNode、Sequential
```

```
type_functor.h
TypeFunctor
```

```
type_relation.h
TypeCallNode、TypeCall、Typereport、TypeReporter、TypeRelationNode、TypeRelation
```

```
type.h
TypeNode、Type、PrimTypeNode、PrimType、PointerTypeNode、PointerType、TypeKind、TypeVarNode、TypeVar、GlobalTypeVarNode、GlobalTypeVar、TupleTypeNode、TupleType、TypeConstraintNode、TypeConstraint、FuncTypeNode、FuncType、IncompleteTypeNode、IncompleteType、RelayRefTypeNode、RelayRefType
```

**node**
```
attr_registry_map.h
AttrRegistryMapContainerMap、AttrRegistryMap
```

```
functor.h
NodeFunctor
```

```
reflection.h
AttrVisitor、ReflectionVTable、ImplVisitAttrs、ImplSEqualReduce、ImplHashReduce、ReflectionTrait、SelectVisitAttrs、SelectEqualReduce、SelectHashReduce
```

```
repr_printer.h
ReprPrinter
```

```
structural_equal.h
BaseValueEqual、StructuralEqual、SEqualReducer
```

```
structural_hash.h
BaseValueHash、StructuralHash、SHashReducer
```

**relay**
```
adt.h
PatternNode、Pattern、PatternWildcardNode、PatternWildcard、PatternVarNode、PatternVar、PatternConstructorNode、PatternConstructor、PatternTupleNode、PatternTuple、ClauseNode、Clause、MatchNode、Match
```

```
base.h
RelayNode、IdNode、Id
```

```
dataflow_matcher.h
DFPatternCallbackNode、DFPatternCallback
```

```
dataflow_pattern_functor.h
DFPatternFunctor、DFPatternVisitor
```

```
dataflow_pattern.h
DFPatternNode、DFPattern、ExprPatternNode、ExprPattern、VarPatternNode、VarPattern、ConstantPatternNode、ConstantPattern、CallPatternNode、CallPattern、FunctionPatternNode、FunctionPattern、LetPatternNode、LetPattern、 TuplePatternNode、TuplePattern、TupleGetItemPatternNode、IfPatternNode、IfPattern、TupleGetItemPattern、AltPatternNode、AltPattern、WildcardPatternNode、WildcardPattern、TypePatternNode、TypePattern、ShapePatternNode、ShapePattern、DataTypePatternNode、DataTypePattern、AttrPatternNode、AttrPattern、DominatorPatternNode、DominatorPattern
```

```
expr_functor.h
ExprFunctor、ExprVisitor、ExprMutator、MixedModeVisitor、MixedModeMutator、ExprRewriter
```

```
expr.h
ConstantNode、Constant、TupleNode、Tuple、VarNode、Var、CallNode、Call、LetNode、Let、IfNode、If、TupleGetItemNode、TupleGetItem、RefCreateNode、RefCreate、RefReadNode、RefRead、RefWriteNode、RefWrite、TempExprNode、TempExpr
```

```
feature.h
Feature、FeatureSet
```

```
function.h
FunctionNode、Function
```

```
interpreter.h
InterpreterClosureObj、InterpreterClosure、RecClosureObj、RecClosure、RefValueObj、RefValue、ConstructorValueObj、ConstructorValue
```

```
op_attr_types.h
OpPatternKind
```

```
op_strategy.h
OpImplementationNode、OpImplementation、OpSpecializationNode、OpSpecialization、OpStrategyNode、OpStrategy
```

```
pattern_functor.h
PatternFunctor、PatternVisitor、PatternMutator
```


**tir**
```
analysis.h
ExprDeepEqual
```

```
buffer.h
BufferType、BufferNode、Buffer、DataProducerNode、DataProducer、Stmt
```

```
data_layout.h
Layout、LayoutNode、LayoutAxis、BijectiveLayoutNode、BijectiveLayout
```

```
function.h
PrimFuncNode、PrimFunc、LinkedParamNode、LinkedParam
```


```
expr_functor.h
ExprFunctor、ExprVisitor、ExprMutator
```

```
expr.h
StringImmNode、StringImm、CastNode、Cast、BinaryOpNode、AddNode、Add、SubNode、Sub、MulNode、Mul、DivNode、Div、ModNode、Mod、FloorDivNode、FloorDiv、FloorModNode、FloorMod、MinNode、Min、MaxNode、Max、CmpOpNode、EQNode、EQ、NENode、NE、LTNode、LT、LENode、LE、GTNode、GT、GENode、GE、AndNode、And、OrNode、Or、NotNode、Not、SelectNode、Select、BufferLoadNode、BufferLoad、ProducerLoadNode、ProducerLoad、LoadNode、Load、RampNode、Ramp、BroadcastNode、Broadcast、LetNode、Let、CallNode、Call、ShuffleNode、Shuffle、CommReducerNode、CommReducer、ReduceNode、Reduce、AnyNode、Any 
```

```
op_attr_types.h
CallEffectKind
```

```
stmt_functor.h
StmtFunctor、StmtVisitor、StmtMutator、StmtExprVisitor、StmtExprMutator
```

```
stmt.h
StmtNode、Stmt、LetStmtNode、LetStmt、AttrStmtNode、AttrStmt、AssertStmtNode、AssertStmt、StoreNode、Store、BufferStoreNode、BufferStore、BufferRealizeNode、BufferRealize、ProducerStoreNode、ProducerStore、ProducerRealizeNode、ProducerRealize、AllocateNode、Allocate、SeqStmtNode、SeqStmt、IfThenElseNode、IfThenElse、EvaluateNode、Evaluate、ForKind、ForNode、For、WhileNode、While、PrefetchNode、Prefetch、BufferRegionNode、BufferRegion、MatchBufferRegionNode、MatchBufferRegion、BlockNode、Block、BlockRealizeNode、BlockRealize
```


**auto_schedule**
```
auto_schedule.h
TuningOptionsNode、TuningOptions
```

```
compute_dag.h
AccessAnalyzerNode、AccessAnalyzer、ComputeDAGNode、LayoutRewriteOption、ComputeDAG
```

```
cost_model.h
CostModelNode、CostModel、RandomModelNode、RandomModel、PythonBasedModelNode、PythonBasedModel、
```

```
loop_state.h
StageKind、ComputeAtKind、StageAttributes、StageNode、Stage、AttachMapNode、AttachMap、StateNode、State
```


```
measure_record.h
RecordToFileNode、RecordToFile、RecordReaderNode、RecordReader
```

```
measure.h
MeasureErrorNO、MeasureInputNode、MeasureInput、BuildResultNode、BuildResult、MeasureResultNode、MeasureResult、MeasureCallbackNode、MeasureCallback、PythonBasedMeasureCallbackNode、PythonBasedMeasureCallback、ProgramBuilderNode、ProgramBuilder、ProgramRunnerNode、ProgramRunner、LocalBuilderNode、LocalBuilder、LocalRunnerNode、LocalRunner、RPCRunnerNode、RPCRunner、ProgramMeasurerNode、ProgramMeasurer
```

```
search_policy.h
SearchCallbackNode、SearchCallback、PreloadMeasuredStatesNode、PreloadMeasuredStates、SearchPolicyKey、SearchPolicyNode、SearchPolicy
```

```
search_task.h
HardwareParamsNode、HardwareParams、SearchTaskNode、SearchTask
```

```
transform_step.h
IteratorKind、IteratorAnnotation、IteratorNode、Iterator、StepNode、Step、AnnotationStepNode、AnnotationStep、FuseStepNode、FuseStep、PragmaStepNode、PragmaStep、ReorderStepNode、ReorderStep、SplitStepNode、SplitStep、FollowSplitStepNode、FollowSplitStep、FollowFusedSplitStepNode、FollowFusedSplitStep、StorageAlignStepNode、StorageAlignStep、ComputeAtStepNode、ComputeAtStep、ComputeInlineStepNode、ComputeInlineStep、ComputeRootStepNode、ComputeRootStep、CacheReadStepNode、CacheReadStep、CacheWriteStepNode、CacheWriteStep、RfactorStepNode、RfactorStep
```





TVM源码
Map方法
at(key)： 通过键访问值，如果键不存在会抛出异常
[key]： 下标操作符
size()： 返回Map中键值对的数量
count(key)： 返回指定键的元素数量
empty()： 检查Map是否为空
clear()： 清空Map，重置为空状态
set(key, value)： 设置键值对
erase(key)： 删除指定键的元素
begin()/end()： 返回指向开始和结束位置的迭代器
find(key)： 查找指定键，返回对应的迭代器
CopyOnWrite()： 实现写时拷贝优化
GetMapNode()： 将基类指针转换为MapNode指针
Merge(Map, Map)： 合并两个Map对象

String方法
compare
c_str()： 返回c风格的字符串指针
size()/length()： 返回字符串字节长度
empty()： 检查字符串是否为空
at()： 字符访问，进行边界检查，越界抛出异常
data()： 返回指向字符数据的指针

Array方法
begin()/end()/rbegin()/rend()： 标准的迭代器接口
[]： 提供数组下标访问语法，进行空指针检查和边界检查
size()： 返回元素个数
capacity()： 返回容器容量
empty()： 检查是否为空
front()/back(): 访问第一个元素/最后一个元素
push_back(item): 在数组末尾添加元素
insert(position, val)： 在指定位置插入元素
insert(position, first, last)： 插入一个范围的元素
pop_back(): 删除最后一个元素
erase(position)： 删除指定位置或范围的元素
resize(): 改变数组大小
reserve(): 预留容量
clear(): 清空数组
Set(i, value)： 设置指定位置的元素值
GetArrayNode()： 安全地获取底层ArrayNode指针
CopyOnWrite()： 写时拷贝

Map<IterVar, Range> te::InferBound(const Schedule& sch)
GraphContext{FeedGraph, AttachPath, std::unordered_map<IterVar, IterVar>, std::unordered_map<const Object*, Stage>}

ScheduleNode: outputs、stages、groups、stage_map、op2stage_cache_
StageNode: op、origin_op、all_iter_vars、leaf_iter_vars、env_threads、store_predicate、relations、iter_var_attrs、attach_type、attach_ivar、attach_stage、scope、is_output、double_buffer、group、num_child_stages
IterVarAttrNode: iter_type、bind_thread、prefetch_data、prefetch_offset、tensor_intrin、dim_align_factor、dim_align_offset、pragma_keys、pragma_values
ReadGraph = Map<Operation, Array<Tensor>>
AttachPath = Map<Opeation, Array<IterVar>>
FeedGraph = std::unordered_map<Tensor, std::vector<Operation>>


InferRootBound
对于输出阶段或占位符操作，直接使用其根迭代变量的定义域，验证域已定义且不重复，然后添加到结果映射中
遍历当前操作的所有输出张量，为每个张量创建对应维度的张量域，从feed图中找到消费这些张量的操作，加入consumers集合
推断当前阶段的存储作用域，获取当前阶段的附加路径
从后向前遍历消费者的叶子迭代变量，检查是否找到附加点，获取该迭代变量的范围，根据不同情况设置up_state
处理消费者的附加路径中的迭代变量，如果需要放松约束，将变量加入relax_set
创建域映射和算术分析器，绑定已知的变量范围，对消费者的根迭代变量计算最终范围，考虑up_state中的约束和relax_set中的放松

InferStorageScope
自动存储作用域推断：数据局部性优化(根据计算的线程层次结构选择合适的存储位置)、内存层次匹配(让存储等级与线程执行等级相匹配)、性能优化(避免不必要的内存访问开销)


InferBound
遍历调度的所有输出操作，构建数据流图
遍历所有计算阶段，构建线程绑定映射，构建操作到阶段的映射
创建每个操作的附加路径



PassDownDomain函数的流程
输入：根迭代变量的范围 → 线程绑定分析 → 正向遍历关系图(split、fuse、rebase、singleton) → 同步线程变量范围 → 输出： 所有迭代变量的精确范围

CreateAttachPath
将高级的compute_at指令转换为具体的执行路径，将每个阶段的相关迭代变量拼接成完整路径，确保路径中变量的顺序反映实际的循环嵌套


te::ScheduleOps
输入: schedule+边界映射 → 初始化和数据准备 → 验证调度正确性 → 反向遍历所有阶段(跳过占位符阶段、获取附加规范、根据附加类型分别处理) → 后处理优化 → 输出: 完整的执行语句

TensorDimKey: Op value_index dim

GetPerStoreFeaturesWorkerFunc
task->compute_dag.ApplySteps(state->transform_steps)
sch.normalize_for_feature_extraction
te::InferBound
te::ScheduleOps
te::VerifyCompactBuffer
tvm::transform::PassContext::Current()
GetBinds
te::SchedulePostProcToPrimFunc
WithAttr
pass_ctx->GetConfig<Bool>("tir.noalias", )
pass_ctx->GetConfig<Bool>("tir.disable_vectorize")
pass_ctx->GetConfig<Bool>("tir.instrument_bound_checkers")
mod = IRModule(Map<GlobalVar, BaseFunc>({{global_var, f}}))
pass_list = Array<tvm::transform::Pass>()
optimize = tir::transform::Sequential(pass_list)
optimize(mod)
it = mod->functions.find(global_var)
prim_func = (*it).second.as<PrimFuncNode>
GetPerStoreFeature(prim_func->body, task->hardware_params->cache_line_bytes, max_n_bufs, feature)



te::SchedulePostProcToPrimFunc函数，负责将调度后处理的结果转换为TIR的原始函数


template <class T>
using BufferMap = std::unordered_map<Buffer, T, ObjectHash, ObjectEqual>;

BufferMap<FeatureSet> buffer_features;


struct FeatureSet {
  // Group 1: Computation related features
  float float_mad;                  // The number of float MAD (Multiply–add) ops
  float float_addsub;               // The number of float add and sub ops
  float float_mul;                  // The number of float multiply ops
  float float_divmod;               // The number of float div and mod ops
  float float_cmp;                  // The number of float comparison ops
  float float_math_func;            // The number of float math func calls
  float float_other_func;           // The number of other float func calls
  float int_mad;                    // The number of integer MAD (Multiply–add) ops
  float int_addsub;                 // The number of integer add and sub ops
  float int_mul;                    // The number of float multiply ops
  float int_divmod;                 // The number of float div and mod ops
  float int_cmp;                    // The number of float comparison ops
  float int_math_func;              // The number of float math func calls
  float int_other_func;             // The number of other float func calls
  float bool_op;                    // The number of bool ops
  float select_op;                  // The number of select ops
  float vec_num;                    // The number of vectorized iterators
  float vec_prod;                   // The product of the lengths of vectorized iterators
  float vec_len;                    // The length of the innermost vectorized iterator
  AnnotationPosType vec_type;       // The type of vectorization position
  float unroll_num;                 // The number of unrolled iterators
  float unroll_prod;                // The product of the lengths of vectorized iterators
  float unroll_len;                 // The length of the innermost unrolled iterator
  AnnotationPosType unroll_type;    // The type of unroll position
  float parallel_num;               // The number of paralleled iterators
  float parallel_prod;              // The product of the lengths of paralleled iterators
  float parallel_len;               // The length of the innermost paralleled iterators
  AnnotationPosType parallel_type;  // The type of parallel position
  float is_gpu;                     // Whether it is a GPU task
  float blockIdx_x_len;             // The length of blockIdx.x
  float blockIdx_y_len;             // The length of blockIdx.y
  float blockIdx_z_len;             // The length of blockIdx.z
  float threadIdx_x_len;            // The length of threadIdx.x
  float threadIdx_y_len;            // The length of threadIdx.y
  float threadIdx_z_len;            // The length of threadIdx.z
  float vthread_len;                // The length of virtual thread

  // Group 2: Buffer access related features (per buffer)
  std::vector<BufferAccessFeature> access_feas;

  // Group 3: Arithmetic intensity related features
  float arith_intensity_curve[ARITH_INTENSITY_CURVE_SAMPLE_N];  // points sampled from the
                                                                // arithmetic intensity curve

  // Group 4: Allocation related features
  float alloc_size;        // The size of allocated buffer in bytes
  float alloc_outer_prod;  // The product of lengths of loops outside the scope of the allocation
  float alloc_inner_prod;  // The product of lengths of loops inside the score of the allocation
  float alloc_prod;        // alloc_outer_prod * alloc_inner_prod

  // Group 5: Outer scope related features
  float outer_prod;            // The product of lengths of outer loops
  float num_loops;             // The number of outer loops
  float auto_unroll_max_step;  // The value of pragma "auto_unroll_max_step"
};


struct BufferAccessFeature {
  std::string buffer_name;        // The name of the buffer
  BufferAccessType acc_type;      // The type of the access
  float bytes;                    // The touched memory in bytes
  float unique_bytes;             // The touched unique memory in bytes
  float lines;                    // The number of touched cache lines
  float unique_lines;             // The number touched unique cache lines
  ReuseType reuse_type;           // Tye type of data reuse
  float reuse_dis_iter;           // The reuse distance in iterator number
  float reuse_dis_bytes;          // The reuse distance in total touched bytes
  float reuse_ct;                 // The reuse ratio
  float bytes_d_reuse_ct;         // bytes / reuse_ct
  float unique_bytes_d_reuse_ct;  // unique_bytes / reuse_ct
  float lines_d_reuse_ct;         // lines / reuse_ct
  float unique_lines_d_reuse_ct;  // unique_lines / reuse_ct
  float stride;                   // The stride in access
};

enum class BufferAccessType : int { kRead = 0, kWrite = 1, kReadWrite = 2, kUnknownRW = 3 };
enum class ReuseType : int { kLoopMultipleRead = 0, kSerialMultipleReadWrite = 1, kNoReuse = 2 };


PerStoreFeatureExtractor extractor(cache_line_size)
extractor(stmt)


PerStoreFeatureExtractor → StmtExprVisitor → StmtVisitor、ExprVisitor
StmtVisitor → StmtFunctor
ExprVisitor → ExprFunctor


StmtVisitor<br>
LetStmtNode: 将一个变量绑定到一个值，然后在绑定的作用域内执行后续语句 <br>
作用: 变量作用域管理(为临时计算结果创建命名变量、管理变量的生命周期和作用域、避免重复计算，提高代码效率)、代码优化(公共子表达式提取、中间结果缓存、代码重组和简化)
```
class LetStmtNode : public StmtNode {
  public:
  Var var;          // 要绑定的变量
  PrimExpr value;   // 绑定的值/表达式
  Stmt body;        // 在绑定作用域内执行的语句体
};

let var = value in
  body

公共子表达式提取
// 原始代码: (a+b)*(a+b)
// 优化后使用LetStmt
let temp = a + b in
  temp * temp
```

AttrStmtNode: 为语句体定义特定的辅助属性，这些属性为IR变换过程提供关键的元信息 <br>
线程管理属性: thread_extent(标记线程启动范围，用于GPU编程)、virtual_thread(标记虚拟线程的启动) <br>
存储管理属性: realize_scope(标记存储作用域)、buffer_bind_scope(缓冲区绑定作用域)、double_buffer_scope(双缓冲作用域)、buffer_dim_align(缓冲区维度对齐信息) <br>
并行和协处理器属性: coproc_scope(标记协处理器处理区域)、coproc_uop_scope(协处理器微操作作用域) <br>
```
class AttrStmeNode: public StmtNode {
  public:
  ObjectRef node;     // 属性关联的节点对象 (如变量、缓冲区、迭代变量等)
  String attr_key;    // 属性的类型键 (如thread_extent、virtual_thread、pipeline_exec_scope、realize_scope等)
  PrimExpr value;     // 属性值
  Stmt body;          // 在该属性作用域内执行的语句体
};

// attr [node] attr_key = value
{
  body    // 在属性作用域内执行
}

// attr [threadIdx.x] thread_extent = 32
for (int i = 0; i < 32; ++i) {
  // 线程并行执行的代码
}

// attr [buffer] realize_scope = "shared"
{
  // 在共享内存作用域内的操作
}
```
AssertStmtNode: 在运行时检查特定条件，如果条件不满足则报告错误信息，然后继续执行后续语句
边界检查、内存对齐验证、GPU代码约束检查、数值范围验证
```
class AssertStmtNode : public StmtNode {
  public:
  PrimExpr condition;   // 要检查的条件
  PrimExpr message;     // 错误时显示的信息
  Stmt body;            // 断言通过后执行的语句体
};

if (!condition) {
  // 报告错误: message
  // 可能终止执行或抛出异常
}
// 继续执行 body
```

StoreNode: 向指定的缓冲区地址写入数据值，支持向量化操作和条件写入
```
class StoreNode : public StmtNode {
  public:
  Var buffer_var;     // 缓冲区变量
  PrimExpr value;     // 要存储的值
  PrimExpr index;     // 存储位置的索引
  PrimExpr predicate; // 存储条件 (掩码)
};

//基本形式
((DType*)buffer_var)[index] = value;

//向量化形式
auto buffer = static_cast<float*>(buffer_var);
buffer[index.v0] = value.v0;
buffer[index.v1] = value.v1;
buffer[index.v2] = value.v2;
```
**BufferRealizeNode的依赖**
无前置依赖:可以独立创建缓冲区、 后续依赖:为后续的BufferStore/BufferLoad提供基础
**BufferStoreNode的依赖**
前置依赖: 必须在对应的BufferRealize作用域内、运行时依赖: 缓冲区必须已经分配内存

BufferStoreNode: 向多缓冲区的指定位置写入数据值，提供高级的、语义化的缓冲区访问接口
```
class BufferStoreNode : public StmtNode {
  public:
  Buffer buffer;            // 高级缓冲区对象
  PrimExpr value;           // 要存储的值
  Array<PrimExpr> indices;  // 多维索引数组
};

buffer[i, j] = value;

矩阵计算
// C[i, j] = A[i, k] * B[k, j]
BufferStore(C, mul_result, {i, j})
```

BufferRealizeNode: 注解缓冲区在特定区域内需要被读写的范围，编译器只需要为相应区域分配内存空间 <br>
BufferRealizeNode最终会被lowered为AllocateNode
```
class BufferRealizeNode : public StmtNode {
  public:
  Buffer buffer;        // 要实现的缓冲区
  Array<Range> bound;   // 要实现的边界范围
  PrimExpr condition;   // 实现条件
  Stmt body;            // 实现体语句
};

buffer_realize buffer_name([min0, extent0], [min1, extent1], ...) if condition {
  body
}

buffer_realize A([0, 100], [0, 50]) {
  // 在A[0:100, 0:50]区域内的操作
}

张量计算优化
// 只为实际时可用的tile分配内存
BufferRealize shared_A([tile_i, tile_size], [tile_k, tile_size]) {
  // shared memory 中的矩阵乘法tile
}
```

ProducerStoreNode: 将值存储到由DataProducer生产的多维数组，供该生产者的消费者读取 <br>
ProducerStore只存在于高级DSL中，不应该出现在有效的TIR PrimFunc中，必须在TIR变换之前被lowered
```
class ProducerStoreNode : public StmtNode {
  public:
  DataProducer producer;    // 数据生产者
  PrimExpr value;           // 要存储的值
  Array<PrimExpr> indices;  // 函数的索引参数
};

高级DSL阶段
// 使用DataProducer (如Tensor)
ProducerStore(tensor_A, computation_result, {i, j})
Lowering后的TIR
// 转换为BufferStore或Store
BufferStore(buffer_A, computation_result, {i, j})
// 或
Store(buffer_A_data, computation_result, flattened_index, predicate)
```

ProducerRealizeNode: 注解数据生产者需要在body中被写入和读取的边界，编译器将为相应区域分配内存空间 <br>
ProducerRealize只存在于高级DSL中，不应该出现在有效的TIR PrimFunc中，必须在TIR变换之前被lowered
```
class ProducerRealizeNode: public StmtNode {
  public:
  DataProducer producer;    // 生产数据的数据生产者
  Region bounds;            // 要实现的边界
  PrimExpr condition;       // 实现的条件
  Stmt body;                // 实现体语句
};

高级DSL阶段
// 使用DataProducer (如Tensor)
ProducerRealize(tensor_A, bounds, condition, body)
Lowering后的TIR
// 转换为BufferRealize
BufferRealize(buffer_A, bounds, condition, body)

张量计算DSL:
// 在Tensor Expression (TE) DSL中
ProducerRealize(output_tensor, output_bounds, condition) {
  // 计算和存储操作
  ProducerStore(output_tensor, computed_value, {i, j})
}
```

AllocateNode: 表示缓冲区的内存分配操作
```
class AllocateNode: public StmtNode {
  public:
  Var buffer_var;           // 缓冲区变量
  DataType dtype;           // 缓冲区的数据类型
  Array<PrimExpr> extents;  // 缓冲区的维度大小
  PrimExpr condition;       // 分配条件(只有满足条件才分配)
  Stmt body;                // 在分配的缓冲区中执行的语句体
}
```

SeqStmtNode: 表示语句序列，用于将多个语句组织成一个有序的执行序列
```
class SeqStmtNode: public StmtNode {
  public:
  Array<Stmt> seq;  // 内部语句序列内容
};
```

IfThenElseNode: 表示条件分支语句
```
class IfThenElseNode: public StmtNode {
  public:
  PrimExpr condition;   // 条件表达式
  Stmt then_case;       // 条件为真时执行的语句
  Stmt else_case;       // 条件为假时执行的语句
};
```

EvaluateNode: 评估一个表达式并忽略其返回值，将表达式转换为语句的桥梁<br>
将call节点转换为语句，使其能够在语句上下文中执行，处理有副作用的表达式，如果表达式没有副作用，节点可以被安全移除
```
class EvaluateNode: public StmtNode {
  public:
  PrimExpr value; //需要被评估的表达式
}
```

ForNode: 表示各种类型的for循环
```
class ForNode: public StmtNode {
  public:
  Var loop_var;                       // 循环变量
  PrimExpr min;                       // 迭代的最小值
  PrimExpr extent;                    // 迭代的范围大小
  ForKind kind;                       // 循环的类型
  Stmt body;                          // 循环体
  Optional<IterVar> thread_binding;   // 线程绑定(仅当kind为kThreadBinding时有效)
  Map<String, ObjectRef> annotations; // 附加注解
}

// for (loop_var = min; loop_var < min + extent; ++loop_var) {
//  body
//}
```


WhileNode: 表示while循环
```
class WhileNode: public StmtNode {
  public:
  PrimExpr condition;   // 终止条件
  Stmt body;            // 循环体
}
```


PrefetchNode: 表示内存预取提示，指导硬件或编译器提前将数据加载到缓存中，以减少内存访问延迟
```
class PrefetchNode: public StmtNode {
  public:
  Buffer buffer;        // 要预取的缓冲区
  Array<Range> bounds;  // 要预取的边界范围
}
```


BufferRegionNode: 表示多维缓冲区的访问区域，用于描述对缓冲区特定区域的访问模式 <br>
精确描述缓冲区的哪个子区域被访问、通过Range数组支持多维缓冲区
```
class BufferRegionNode: public object {
  public:
  Buffer buffer;        // 缓冲区引用
  Array<Range> region;  // 区域范围数组，每个维度一个Range
}

# 目标代码
# 访问矩阵A的子区域: A[0:4, 2:6]
# 访问矩阵B的子区域: B[2:6, 1:5]
TIR表示
Buffer A = decl_buffer({8, 8}, DataType::Float(32), "A")
Buffer B = decl_buffer({8, 8}, DataType::Float(32), "B")

Array<Range> A_region = {
  Range::FromMinExtent(0, 4),
  Range::FromMinExtent(2, 6)
}
BufferRegion A_access(A, A_region)

Array<Range> B_region = {
  Range::FromMinExtent(0, 4),
  Range::FromMinExtent(1, 5)
}
BufferRegion B_access(B, B_region)

BufferRegion A_full = BufferRegion::FullRegion(A)
```

MatchBufferRegionNode: 表示缓冲区映射约束
```
class MatchBufferRegionNode: public Object {
  public:
  Buffer buffer;
  BufferRegion source;
}
```

BlockNode: 表示一个独立的计算块
```
class BlockNode: public StmtNode {
  public:
  Array<IterVar> iter_var;                  // 块的迭代变量
  Array<BufferRegion> reads;                // 读取的缓冲区区域
  Array<BufferRegion> writes;               // 写入的缓冲区区域
  String name_hint;                         // 块的名称提示
  Stmt body;                                // 块的主体语句
  Optional<Stmt> init;                      // 初始化语句
  Array<Buffer> alloc_buffers;              // 在块中分配的缓冲区
  Array<MatchBufferRegion> match_buffers;   // 匹配的缓冲区区域
  Map<String, ObjectRef> annotations;       // 块的注解
}
```

BlockRealizeNode: 在特定绑定值下执行Block的实现节点
```
class BlockRealizeNode : public StmtNode {
  public:
  Array<PrimExpr> iter_values;  // 迭代变量的对应值
  PrimExpr predicate;           // Block实现的谓词条件
  Block block;                  // 要被实现的Block
}
```


ExprVisitor
VarNode、AnyNode
SizeVarNode、
LoadNode、
BufferLoadNode、ProducerLoadNode、
LetNode、
CallNode、
AddNode、SubNode、MulNode、DivNode、ModNode、FloorDivNode、FloorModNode、MinNode、MaxNode、EQNode、NENode、LTNode、LENode、GTNode、GENode、AndNode、OrNode、
ReduceNode、
CastNode、
NotNode、
SelectNode、
RampNode、
BroadcastNode、
ShuffleNode、
IntImmNode、FloatImmNode、StringImmNode


placeholder = PLACEHOLDER [1, 2048]
placeholder = PLACEHOLDER [1000, 2048]
T_dense(i, j) += (placeholder[i, k]*placeholder[j, k])
placeholder = PLACEHOLDER [1000]
T_add(ax0, ax1) = (T_dense[ax0, ax1] + placeholder[ax1])


特征提取工作
ExtractComputationFeature
计算操作特征(统计各种数学运算的总执行次数、区分浮点、整数、布尔运算类型)、循环优化特征(向量化、展开、并行化)、GPU特征(GPU执行的线程组织信息、用于GPU代码生成和优化)
float: mad、addsub、mul、divmod、cmp、math_func、other_func
int: mad、addsub、mul、divmod、cmp、math_func、other_func
bool_op、select_op
vec_len、unroll_len、parallel_len
vec_type、unroll_type、parallel_type
vec_num、unroll_num、parallel_num
is_gpu、blockIdx_x_len、blockIdx_y_len、blockIdz_z_len、threadIdx_x_len、threadIdx_y_len、threadIdx_z_len、vthread_len

ExtractBufferAccessFeature
分析缓冲区的访问模式(缓冲区名称、访问类型、步长、访问字节数、缓存行数、重用类型、距离、次数)
access_feas

ExtractArithmeticIntensityFeature
计算算术强度曲线，衡量计算密集度，算术强度=计算操作数(FLOPS)/内存访问字节数(Bytes)
arith_intensity_curve

ExtractOuterScopeFeature
计算执行规模、循环嵌套的深度、循环展开的配置
alloc_size、alloc_prod、alloc_outer_prod、alloc_inner_prod

ExtractAllocationFeature
提取与内存分配相关的特征(缓冲区占用的内存字节数、缓冲区分配的总工作量、分配点外层循环的规模、分配点内层循环的规模)
outer_prod、num_loops、auto_unroll_max_step


ReuseTypePAM:
返回值: 重用类型、重用距离-迭代器、重用距离-字节数、重用计数


Analyzer::Bind
Canonical Simplification(标准化化简)：将表达式转换为标准形式 eg: 将x+1+2简化为x+3;将2*x+3*x简化为5*x
Rewrite Simplification(重写简化): 基于模式匹配和重写规则的简化 eg: 将x*0简化为0; 将x+0简化为x; 将x/1简化为x
Const Int Bound(更新常量整数边界分析器): 将变量var的整数边界信息更新为表达式的边界 eg: expr=x+5且x的范围是[0,10],则var的边界更新为[5, 15]
Modular set(更新模运算集合分析器): 将变量var的模运算集合信息更新为表达式的模信息 模运算集合({coeff * x + base | x ∈ Z}) eg: expr = 4 * x,则var的模信息为系数4，基数0
Rewrite Simplify(更新重写简化器): 在重写简化器中记录变量var绑定到new_var,后续简化过程中，遇到var时可以直接替换为new_var
Canonical Simplify(更新标准化简化器): 在标准化简化器中记录变量var绑定到new_var,缓存变量的标准化表示，避免重复标准化计算


```
auto pass_list = Array<tvm::transform::Pass>();
// Phase 0
pass_list.push_back(tir::transform::InjectPrefetch());
pass_list.push_back(tir::transform::StorageFlatten(64, instrument_bound_checkers));
// Phase 1
pass_list.push_back(tir::transform::NarrowDataType(32));
pass_list.push_back(tir::transform::Simplify());
pass_list.push_back(tir::transform::VectorizeLoop(!disable_vectorize));
pass_list.push_back(tir::transform::InjectVirtualThread());
pass_list.push_back(tir::transform::StorageRewrite());
pass_list.push_back(tir::transform::Simplify());
tvm::Map<String, tvm::PrimExpr> gpu_params{
    {"max_shared_memory_per_block", task->hardware_params->max_shared_memory_per_block},
    {"max_local_memory_per_block", task->hardware_params->max_local_memory_per_block},
    {"max_threads_per_block", task->hardware_params->max_threads_per_block},
    {"max_vector_bytes", task->hardware_params->vector_unit_bytes},
    {"max_vthread", task->hardware_params->max_vthread_extent},
};
pass_list.push_back(tir::transform::VerifyGPUCode(gpu_params));
const auto& optimize = tir::transform::Sequential(pass_list);
optimize(mod);
```
上面这段代码的详解:
需经历的代码文件: tir/transform.h、tir/transfroms/文件夹底下对应pass的.cc文件
tir::transform::Sequential(pass_list) 对应 ir/tranforms.h → ir/transform.cc
每个pass里面都有CreatePrimFuncPass函数 在/tir/ir/transform.cc 经历过程 CreatePrimFuncPass → PrimFuncPass::PrimFuncPass → PrimFuncPassNode::operator() 注意在这里面pass_func传递过来的是一个函数
然后经历每个每个pass，访问Stmt



TVM里面某个变量是ObjectRef类型，但实际上它有具体的类型，可采用下面的方法进行gdb调试查看
假设你有一个AttrStmtNode* 指针，命名为 attr_stmt
先检查node的实际类型
p attr_stmt->node.get()->GetTypeKey()
或者检查类型索引
p attr_stmt->node.get()->type_index_

$85 = {static npos = 18446744073709551615, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0x7cd56f3fd780 "tir.IterVar"}, _M_string_length = 11, {_M_local_buf = "tir.IterVar\000\325|\000", 
    _M_allocated_capacity = 8243122550632245620}}

将ObjectRef转换为IterVar
p ((tvm::tir::IterVarNode*)attr_stmt->node.get())->var->name_hint.c_str()



ir/transform.cc:242
IRModule Pass::operator()(IRModule mod) const {
  const PassNode* node = operator->();
  ICHECK(node != nullptr);
  PassProfile::EnterPass(node->Info()->name);
  auto ret = node->operator()(std::move(mod));
  PassProfile::ExitPass();
  return std::move(ret);
}


ir/transform.h:314
PassNode
IRModule operator()(IRModule mod) const {
  return this->operator()(std::move(mod), PassContext::Current());
}


ir/transform.cc:531
IRModule SequentialNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  for (const Pass& pass : passes) {
    ICHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!pass_ctx.PassEnabled(pass_info)) continue;
    // resolve dependencies
    for (const auto& it : pass_info->required) {
      mod = GetPass(it)(std::move(mod), pass_ctx);
    }
    mod = pass(std::move(mod), pass_ctx);
  }
  return mod;
}

ir/transform.cc:251
IRModule Pass::operator()(IRModule mod, const PassContext& pass_ctx) const {
  const PassNode* node = operator->();
  ICHECK(node != nullptr);
  PassProfile::EnterPass(node->Info()->name);
  auto ret = node->operator()(std::move(mod), pass_ctx);
  PassProfile::ExitPass();
  return std::move(ret);
}


tir/ir/transform.cc:89
IRModule PrimFuncPassNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  const PassInfo& pass_info = Info();
  ICHECK(mod.defined());
  pass_ctx.Trace(mod, pass_info, true);
  std::vector<ObjectRef> deleted_list;
  IRModuleNode* mod_ptr = mod.CopyOnWrite();
  auto* func_dict = mod_ptr->functions.CopyOnWrite();
  // directly loop over the underlying dict
  for (auto& kv : *func_dict) {
    // only picks up tir::PrimFunc
    if (kv.second->IsInstance<PrimFuncNode>()) {
      // move out the function so that it is the only copy.
      PrimFunc func = Downcast<PrimFunc>(std::move(kv.second));
      func = pass_func(std::move(func), mod, pass_ctx);
      kv.second = std::move(func);

      if (!kv.second.defined()) {
        deleted_list.push_back(kv.first);
      }
    }
  }

  // automatic removal of None
  for (const auto& gv : deleted_list) {
    func_dict->erase(gv);
  }
  pass_ctx.Trace(mod, pass_info, false);
  return mod;
}


packed_func.h:1505
template <typename R, typename... Args>
TVM_ALWAYS_INLINE R TypedPackedFunc<R(Args...)>::operator()(Args... args) const {
  return detail::typed_packed_call_dispatcher<R>::run(packed_, std::forward<Args>(args)...);
}


Copy-on-Write 写时复制: (多个变量可以共享同一份数据，只有在需要修改时才创建副本)






C[32,1000]=A[32,2048]*B[2048,1000]




p0            p1
|             |
p0.shared     p1.shared
      |        | 
      |        |
      ——————————
          |
    T.matmul.nn.local
          |
      T.matmul.nn

cache_write
T.matmul.nn.local
split*4 i0.c.inner、i0.c.outer.inner、i0.c.outer.outer.inner、i0.c.outer.outer.outer.inner、i0.c.outer.outer.outer.outer
split*4 i1.c.inner、i1.c.outer.inner、i1.c.outer.outer.inner、i1.c.outer.outer.outer.inner、i1.c.outer.outer.outer.outer
split*2 k.inner、k.outer.inner、k.outer.outer
reorder

i0.c.outer.outer.outer.outer
i1.c.outer.outer.outer.outer
i0.c.outer.outer.outer.inner
i1.c.outer.outer.outer.inner
------------------------------
i0.c.outer.outer.inner
i1.c.outer.outer.inner
k.outer.outer
k.outer.inner
i0.c.outer.inner
i1.c.outer.inner
k.inner
i0.c.inner
i1.c.inner

T.matmul.nn
split*3   i0.inner、i0.outer.inner、i0.outer.outer.inner、i0.outer.outer.outer
split*3   i1.inner、i1.outer.inner、i1.outer.outer.inner、i1.outer.outer.outer

i0.outer.outer.outer  blockIdx.x
i1.outer.outer.outer
i0.outer.outer.inner  vthread
i1.outer.outer.inner
i0.outer.inner      threadIdx.x
i1.outer.inner
------------------------
i0.inner  
i1.inner

compute_at

cache_read
compute_at

cache_read
compute_at

fuse
Annotation

fuse
annotation

fuse
annotation

fuse(p1.shared) ~~ax0.ax1.fused~~
split(p1.shared)  ax0.ax1.fused.inner、~~ax0.ax1.fused.outer~~
Annotation vectorize
followFusedSplit  ax0.ax1.fused.outer.inner、ax0.ax1.fused.outer.outer
Annotation kThreadx

fuse(p0.shared) ~~ax0.ax1.fused~~
split(p0.shared)  ax0.ax.fused.inner、~~ax0.ax1.fused.outer~~
Annotation vectorize
followFusedSplit  ax0.ax1.fused.outer.inner、ax0.ax1.fused.outer.outer
Annotation kThreadx

Pragma


BufferRealize   T.matmul.nn (32, 1000)
AttrStmt    blockIdx.x     16=4*4 
            vthread        5=1*5
            threadIdx.x    100=2*50
BufferRealize   T.matmul.nn.local (32, 1000)
SeqStmt
- AttrStmt   pragma           512
  AttrStmt   pragma_explict  
  SeqStmt
  - For           i0.c.outer.inner.init   4
    BufferStore   T.matmul.nn.local             (blockIdx.x%4)*250+(vthread*50)+threadIdx.x%50    (blockIdx//4)*8+(threadIdx.x//50)*4+io.c.outer.inner.init
    
  - For           k.outer.outer     64
    BufferRealize p0.shared
    SeqStmt
    - For ax0.ax1.fused.outer.outer   3
      AttrStmt      threadIdx.x       100
      IfThenElse
      BufferStore   p0.shared     (blockIdx.x//4)*8+(ax0.ax1.fused.outer.outer*100+threadIdx.x)//32 (k.outer.outer*32)+((ax0.ax1.fused.outer.outer*4)+threadIdx.x)%32
    - BufferRealize p1.shared
      SeqStmt
      - For ax0.ax1.fused.outer.outer   80
        AttrStmt  threadIdx.x           100
        BufferStore p1.shared    (k.outer.outer*32)+((ax0.ax1.fused.outer.outer*100)+threadIdx.x)//250  (blockIdx.x%4)*250+((ax0.ax1.fused.outer.outer*100)+threadIdx.x)%250       
      - For k.outer.inner       32
        For i0.c.outer.inner    4
        BufferStore   T.matmul.nn.local   (blockIdx.x//4)*8+(threadIdx.x//50)*4+i0.c.outer.inner   (blockIdx.x%4)*250+vthread*50+threadIdx.x%50
                                          p0.shared   (blockIdx.x//4)*8+(threadIdx.x//50)*4+i0.c.outer.inner    k.outer.outer*32+k.outer.inner
                                          p1.shared   k.outer.outer*32+k.outer.inner    (blockIdx.x%4)*250+vthread*50+threadIdx%50
- For i0.inner  4
  BufferStore T.matmul.nn   (blockIdx.x//4)*8+(threadIdx.x//50)*4+i0.inner    (blockIdx.x%4)*250+(vthread*50)+(threadIdx.x%50)


  4*2048=8192

---------------------------
```
  BufferRealize   T.matmul.nn (32, 1000)
  AttrStmt blockIdx.x   2=1*2
  AttrStmt vthread      8=4*2
  AttrStmt threadIdx.x  100=2*50
  BufferRealize   T.mtmul.nn.local (32, 1000)
  SeqStmt
  - AttrStmt  pragma  512
    AttrStmt  pragma_explicit
    SeqStmt
    - For     i1.c.outer.inner.init   5
      For     i0.c.inner.init         4
      BufferStore   T.matmul.nn.local   (32, 1000)    (vthread//2)*8+(threadIdx.x//50)*4+i0.c.inner.init    (blockIdx.x*500)+(vthread%2)*250+(threadIdx.x%50)*5+i1.c.outer.inner.init
                              T.matmul.nn.local(kWrite)
      i0.c.inner.init                 4*1
      i1.c.outer.inner.init           4*5
      threadIdx.x                     8*250
      vthread                         32*500
      blockIdx.x                      32*1000
      T.matmul.nn.local   kNoReuse   0    0    0    stride=1

    - For     k.outer.outer     128
      BufferRealize p0.shared (32, 2048)
      SeqStmt
      - For ax0.ax1.fused.outer.outer   6
        AttrStmt  threadIdx.x           100
        IfThenElse
        BufferStore p0.shared   (32, 2048)      ((ax0.ax1.fused.outer.outer*100)+threadIdx.x)//16    (k.outer.outer*16)+(ax0.ax1.fused.outer.outer*4+threadIdx.x)%16
                                        p0(kRead)     p0.shared(kWrite)        
        threadIdx.x                   7*16              7*16
        ax0.ax1.fused.outer.outer     38*16             38*16
        k.outer.outer                 38*2048           38*2048
        threadIdx.x                   38*2048           38*2048
        vthread                       38*2048           38*2048
        blockIdx.x                    38*2048           38*2048  
        p0        kLoopMultipleRead   76800   622592(38*2048*4*2)    100    stride=1
        p0.shared kLoopMultipleRead   76800   622592(38*2048*4*2)    100    stride=1


      - BufferRealize p1.shared   (2048, 1000)
        SeqStmt
        - For  ax0.ax1.fused.outer.outer  80
          AttrStmt  threadIdx.x           100
          BufferStore   p1.shared   (2048, 1000)  (k.outer.outer*16)+(ax0.ax1.fused.outer.outer//5)   (blockIdx.x*50)+(ax0.ax1.fused.outer.outer%5)*100+threadIdx.x
                                        p1(kRead)       p1.shared(kWrite)
          threadIdx.x                   1*100             1*100
          ax0.ax1.fused.outer.outer     16*500            16*500
          k.outer.outer                 2048*500          2048*500
          threadIdx.x                   2048*500          2048*500
          vthread                       2048*500          2048*500
          blockIdx.x                    2048*1000         2048*1000

        - For  k.outer.inner      4
          For  i1.c.outer.inner   5
          For  k.inner            4
          For  i0.c.inner         4
          BufferStore   T.matmul.nn.local (32, 1000)
                        T.matmul.nn.local      (vthread//2)*8+(threadIdx.x//50)*4+i0.c.inner    (blockIdx.x*500)+(vthread%2)*250+(blockIdx.x%50)*5+i1.c.outer.inner
                        p0.shared              (k.outer.outer*16)+(k.outer.inner*4)+k.inner     (blockIdx.x*500)+(vthread%2)*250+(threadIdx.x%50)*5+i1.c.outer.inner
                        p1.shared              (vthread//2)*8+(threadIdx.x//50)*4+i0.c.inner    (k.outer.outer*16)+(k.outer.inner*4)+k.inner
                                p0.shared(kRead)      p1.shared(kRead)      T.matmul.nn.local(kReadWrite)      
          i0.c.inner            1*1                   4*1                   4*1
          k.inner               4*1                   4*4                   4*1
          i1.c.outer.inner      4*5                   4*4                   4*5
          k.outer.inner         16*5                  4*16                  4*5
          k.outer.outer         2048*5                4*2048                4*5
          threadIdx.x           2048*250              8*2048                8*250
          vthread               2048*500              32*2048               32*500
          blockIdx.x            2048*1000             32*2048               32*1000

  - For i0.inner  4
    For i1.inner  5
    BufferStore   T.matmul.nn   (32, 1000)    (vthread//2)*8+(threadIdx.x//50)*4+i0.inner     (blockIdx.x*500)+(vthread%2)*250+(threadIdx.x%50)*5+i1.inner
                    T.matmul.nn.local(kRead)        T.matmul.nn(kWrtie)
    i1.inner        1*5                             1*5
    i0.inner        4*5                             4*5
    threadIdx.x     8*250                           8*250
    vthread         32*500                          32*500
    blockIdx.x      32*1000                         32*1000
```