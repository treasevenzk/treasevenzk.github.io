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