---
layout:     post
title:      HeronCode
subtitle:   Code Reproduction
date:       2025-01-12
author:     Treaseven
header-img: img/bg20.jpg
catalog: true
tags:
    - debug
---

### CPU部分的实验

python run.py -p dlboost -c quick_start.json

程序运行经过历程:<br>
HeronRunner.LocalRunner &rarr; autotvm.measure_option &rarr; op_name(G1),case(["gemm", [64, 64, 64]]) &rarr; 

config配置的属性: <br>
out_name、opt_method、max_trials、runner_number、runner_repeat、runner_timeout、build_timeout、in_dtype、out_dtype、cases、temperature <br>
device_id、tuned、target_name、codegen_type、get_op

Env类 <br>
config: out_name、opt_method、max_trials、runner_number、runner_repeat、runner_timeout、build_timeout、in_dtype、out_dtype、cases、temperature、
device_id、tuned、target_name、codegen_type、get_op、task_name
runner: Runner类
num_ops: len(all_opt_methods)
build_kwargs:
task: