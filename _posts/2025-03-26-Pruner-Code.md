---
layout:     post
title:      PrunerCode
subtitle:   Code Reproduction
date:       2025-03-26
author:     Treaseven
header-img: img/bg20.jpg
catalog: true
tags:
    - debug
---
on-line cost model
Pruner: 1.69ms      Used time 5563s     Estimated total latency: 1.476ms
MoA-Pruner: 1.70ms  Used time 4978s     Estimated total latency: 1.457ms
Ansor: 3.96ms       Used time 6691s     Estimated total latency: 1.592ms


off-line cost model
Pruner-offline: 1.71ms Used time 4212s  Estimated total latency: 1.469ms
TensetMLP:  1.79ms     Used time 5469s  Estimated total latency: 1.621ms

inlcude
tvm/auto_scheduler
feature_pam.h、feature_psa.h

python
tvm/auto_scheduler
cost_model/pam_model.py、psa_model.py

src
auto_scheduler
feature_pam.cc、feature_psa.cc