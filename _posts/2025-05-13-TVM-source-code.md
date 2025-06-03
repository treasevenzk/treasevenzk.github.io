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


$5 = (const tvm::auto_scheduler::FeatureSet &) @0x71eaa400c0c0: {float_mad = 0, float_addsub = 1024, float_mul = 0, float_divmod = 0, float_cmp = 0, float_math_func = 0, float_other_func = 0, int_mad = 0, int_addsub = 2048, int_mul = 2048, int_divmod = 0, int_cmp = 0, int_math_func = 0, int_other_func = 0, bool_op = 0, select_op = 0, vec_num = 0, vec_prod = 0, vec_len = 0, vec_type = tvm::auto_scheduler::AnnotationPosType::kPosNone, unroll_num = 0, unroll_prod = 0, unroll_len = 0, 
  unroll_type = tvm::auto_scheduler::AnnotationPosType::kPosNone, parallel_num = 0, parallel_prod = 0, parallel_len = 0, parallel_type = tvm::auto_scheduler::AnnotationPosType::kPosNone, is_gpu = 1, blockIdx_x_len = 32, blockIdx_y_len = 1, blockIdx_z_len = 1, 
  threadIdx_x_len = 32, threadIdx_y_len = 1, threadIdx_z_len = 1, vthread_len = 1, access_feas = {<std::_Vector_base<tvm::auto_scheduler::BufferAccessFeature, std::allocator<tvm::auto_scheduler::BufferAccessFeature> >> = {
      _M_impl = {<std::allocator<tvm::auto_scheduler::BufferAccessFeature>> = {<__gnu_cxx::new_allocator<tvm::auto_scheduler::BufferAccessFeature>> = {<No data fields>}, <No data fields>}, <std::_Vector_base<tvm::auto_scheduler::BufferAccessFeature, std::allocator<tvm::auto_scheduler::BufferAccessFeature> >::_Vector_impl_data> = {_M_start = 0x71eaa403c010, _M_finish = 0x71eaa403c118, _M_end_of_storage = 0x71eaa403c118}, <No data fields>}}, <No data fields>}, arith_intensity_curve = {0.582413673, 0.582413673, 0.582413673, 
    0.582413673, 0.582413673, 0.613152564, 0.643891394, 0.674630284, 0.705369115, 0.736108005}, alloc_size = 4000, alloc_outer_prod = 1, alloc_inner_prod = 1024, alloc_prod = 1000, outer_prod = 1024, num_loops = 2, auto_unroll_max_step = 0}


PerStoreFeatureExtractor extractor(cache_line_size)
extractor(stmt)


PerStoreFeatureExtractor → StmtExprVisitor → StmtVisitor、ExprVisitor
StmtVisitor → StmtFunctor
ExprVisitor → ExprFunctor


StmtVisitor
LetStmtNode、AttrStmtNode
ForNode、
WhileNode、
AllocateNode、
StoreNode、
BufferStoreNode、
BufferRealizeNode、
IfThenElseNode、
AssertStmtNode、
ProducerStoreNode、
ProducerRealizeNode、
PrefetchNode、
SeqStmtNode、
EvaluateNode、
BlockNode、
BlockRealizeNode


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