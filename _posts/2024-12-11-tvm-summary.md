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
* compilation and execution
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
```