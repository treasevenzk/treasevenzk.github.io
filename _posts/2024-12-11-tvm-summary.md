---
layout:     post
title:      TVM OSDI 2018
subtitle:   TVM An Automated End-to-End Optimizing Compiler for Deep Learning
date:       2024-12-11
author:     Treaseven
header-img: img/bg18.jpg
catalog: true
tags:
    - Tensor Expression Language
    - Automated Program optimization Framework
    - AI compiler
---


### 源码阅读笔记

#### Get started
##### Vector Add
* define the tvm computation
* creating a schedule
* compilation and execution<br>

```
# 原始程序
import numpy as np
np.random.seed(0)
n = 100
a = np.random.normal(size=n).astype(np.float32)
b = np.random.normal(size=n).astype(np.float32)
c = a + b


# 使用tvm
import tvm
for tvm import te # te stands for tensor expression
# Define the TVM compuation
def vector_add(n):
    A = te.placeholder((n,), name='a')
    B = te.placeholder((n,), name='b')
    C = te.compute(A.shape, lambda i: A[i] + B[i], name='c')
    return A, B, C
A, B, C = vector_add(n)
# Creating a schedule
s = te.create_schedule(C.op)
tvm.lower(S, [A, B, C], simple_mode=True)
# Compilation and execution
mod = tvm.build(s, [A, B, C])  # compiles to machine codes
a, b, c = get_abc(100, tvm.nd.array)
mod(a, b, c)
np.testing.assert_array_equal(a.asnumpy() + b.asnumpy(), c.asnumpy())
```

##### Neural Network Inference
relay module in TVM to convert and optimize a neural network. Relay is the high-level intermediate representation (IR) in TVM to represent a neural network.

```
import numpy as np
import mxnet as mx
from PIL import Image
import tvm
from tvm import relay

model = mx.gluon.model_zoo.vision.resnet18_v2(pretrained=True)
# Pre-processing Data
image = Image.opern('../data/cat.jpg').resize((224, 224))
def image_preprocessing(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image.astype('float32')
x = image_preprocessing(image)
# Compile Pre-trained Models 目前仅支持静态图操作，不支持动态图
relay_mod, relay_params = relay.frontend.from_mxnet(model, {'data', x.shape})
target = 'llvm'
with relay.build_config(opt_level=3):
    graph, mod, params = relay.build(relay_mod, target, params=relay_params)  # graph描述神经网络、mod代表编译算子、params代表权重参数
# Inference
ctx = tvm.context(target)
rt = tvm.contrib.graph_runtime.create(graph, mod, ctx)
rt.set_input(**params)
rt.run(data=tvm.nd.array(x))
scores = rt.get_output(0).asnumpy()[0]
# Saving the compiled library
name = 'resnet18'
graph_fn, mod_fn, params_fn = [name+ext for ext in ('.json', '.tar', '.params')]
mod.export_library(mod_fn)
with open(graph_fn, 'w') as f:
    f.write(graph)
with open(params_fn, 'wb') as f:
    f.write(relay.save_param_dict(params))
loaded_graph = open(graph_fn).read()
loaded_mod = tvm.runtime.load_module(mod_fn)
loaded_params = open(params_fn, "rb").read()
loaded_rt = tvm.contrib.graph_runtime.create(loaded_graph, loaded_mod, ctx)
loaded_rt.load_params(loaded_params)
loaded_rt.run(data=tvm.nd.array(x))
loaded_scores = loaded_rt.get_output(0).asnumpy()[0]
tvm.testing.assert_allclose(loaded_scores, scores)
```

#### Expression for Operators
##### Data Types
```
import tvm
from tvm import te
import numpy as np

n = 100

def tvm_vector_add(dtype):
    A = te.placeholder((n,), dtype=dtype)
    B = te.placeholder((n,), dtype=dtype)
    C = te.compute(A.shape, lambda i: A[i] + B[i])
    print('expressiuon dtype:', A.dtype, B.dtype, C.dtype)
    s = te.create_schedule(C.op)
    return tvm.build(s, [A, B, C])

def test_mod(mod, dtype):
    a, b, c = d2ltvm.get_abc(n, lambda x: tvm.nd.array(x.astype(dtype)))
    mod(a, b, c)
    np.testing.asser_equal(c.asnumpy(), a.asnumpy() + b.asnumpy())

for dtype in ['float16', 'float64', 'int8', 'int16', 'int64']:
    mod = tvm_vector_add(dtype)
    test_mod(mod, dtype)
``` 

##### Converting Elements Data Types



#### 常用函数
```tvm.te.placeholder```: declare the placeholders A and B for both inputs by specifying theirs shapes  <br>
```tvm.compute```: compute <br>
```tvm.create_schedule```: how to execute the program, for example, the order to access data and how to do multi-threading parallelization  <br>
```tvm.lower```:  <br>
```tvm.build```: compile them into an executable module <br>
```tvm.nd.array```: convert data type  <br>
```relay_mod, relay_params = relay.fronted.from_mxnet(model {'data': x.shape})```:  <br>
```graph, mod, params = relay.build(relay_mod, target, params=relay_params)```:  <br>
```ctx = tvm.context(target)```:  <br>
```rt = tvm.contrib.graph_runtime.create(graph, mod, ctx)```:  <br>
```rt.set_input(**params)```:  <br>
```rt.run(data=tvm.nd.array(x))```:  <br>
```te.var```: create a symbolic variable for an int32 scalar  <br>
```tvm.reduce_axis```: create an axis for reduction with range from 0 to m  <br>
```tvm.comm_reducer```: a customized commutative reduction operator  <br>


### 官方教程

#### Design and Architecture
```Overall Flow```: model creation、transformation、target translation、runtime execution

<img width="500" height="300" src="/img/post-tvm-design.png"/>

* ```Key data structures```: IRModule(realx::Function、tir::PrimFunc)

* ```Transformations```: relax transformations(common graph-level optimizations)、tir transformations(TensorIR schedule、Lowering Passes)、cross-level transformations

* ```Targete Translation```
* ```Runtime Execution```
