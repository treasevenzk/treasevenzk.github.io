---
layout:     post
title:      Reading List
subtitle:   Compiler Optimization
date:       2024-12-17
author:     Treaseven
header-img: img/bg29.jpg
catalog: true
tags:
    - Machine Learning
    - Deep Learning
    - AI Compiler
---

### Paper
---
Survery
- <img src="https://img.shields.io/badge/24-pages-green.svg" alt="24-pages" align="left"> [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/pdf/2002.03794) - Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan, Guangwen Yang, Depei Qian, IEEE Transactions on Parallel and Distributed Systems, 2021

Dense Tensor Program Optimization
- <img src="https://img.shields.io/badge/6-pages-green.svg" alt="6-pages" align="left"> [A Holistic Functionalization Approach to Optimizing Imperative Tensor Programs in Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3649329.3658483) - Jinming Ma, Xiuhong Li, Zihan Wang Xingcheng Zhang, Shengen Yan, Yuting Chen, Yueqian Zhang, Minxi Jin, Lijuan Jiang, Yun (Eric) Liang, Chao Yang, Dahua Lin, DAC, 2024

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [A Learned Performance Model For Tensor Processing Units](https://proceedings.mlsys.org/paper_files/paper/2021/file/6bcfac823d40046dca25ef6d6d59cc3f-Paper.pdf) - Samuel J.Kaufman, Phitchaya Mangpo Phothilimthana, Yanqi Zhou, Charith Mendis, Sudip Roy, Amit Sabne, Mike Burrows, MLSys, 2021

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2022_2023/papers/ZHENG_ASPLOS_2020.pdf) - Size Zheng, Yun Liang, Shuo Wang, Renze Chen, Kaiwen Sheng, ASPLOS, 2020

- <img src="https://img.shields.io/badge/45-pages-green.svg" alt="45-pages" align="left"> [Modeling the Interplay between Loop Tiling and Fusion in Optimizing Compilers Using Affine Relations](https://dl.acm.org/doi/pdf/10.1145/3635305) - Jie Zhao, Jinchen Xu, Peng Di, Wang Nie, Jiahui Hu, Yanzhi Yi, Sijia Yang, Zhen Geng, Renwei Zhang, Bojie Li, Zhiliang Gan, and Xuefeng Jin, TOCS, 2024

- <img src="https://img.shields.io/badge/19-pages-green.svg" alt="19-pages" align="left"> [Apollo: Automatic Partition-based Operator Fusion through Layer by Layer Optimization](https://proceedings.mlsys.org/paper_files/paper/2022/file/e175e8a86d28d935be4f43719651f86d-Paper.pdf) - Jie Zhao, Xiong Gao, Ruijie Xia, Zhaochuang Zhang, Deshi Chen, Lei Chen, Renwei Zhang, Zhen Geng, Bin Cheng, and Xuefeng Jin, MLSys, 2024

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Optimizing the Memory Hierarchy by Compositing Automatic Transformations on Computations and Data](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9251965) - Jie Zhao, Peng Di, MICRO, 2020

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Tensor Program Optimization with Probabilistic Programs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9251965) - Junru Shao, Xiyou Zhou, Siyuan Feng, Bohan Hou, Ruihang Lai, Hongyi Jin, Wuwei Lin, Masahiro Masuda, Cody Hao Yu, Tianqi Chen, NIPS, 2022

- <img src="https://img.shields.io/badge/43-pages-green.svg" alt="43-pages" align="left"> [Composable and modular code generation in MLIR: A structured and retargetable approach to tensor compiler construction](https://arxiv.org/pdf/2202.03293) - Vasilache Nicolas, Zinenko Oleksandr, Bik Aart JC, Ravishankar Mahesh, Raoux Thomas, Belyaev  Alexander, Springer Matthias, Gysi Tobias, Caballero Diego, Herhut Stephan, Laurenzo Stella, Cohen Albert, arxiv, 2022

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion](https://dl.acm.org/doi/pdf/10.1145/3453483.3454083) - Wei Niu, Jiexiong Guan, Yanzhi Wang, Gagan Agrawal, Bin Ren, PLDI, 2021

- <img src="https://img.shields.io/badge/18-pages-green.svg" alt="18-pages" align="left"> [EINNET: Optimizing Tensor Programs with Derivation-Based Transformations](https://www.usenix.org/system/files/osdi23-zheng.pdf) - Liyan Zheng, Haojie Wang, Jidong Zhai, Muyan Hu, Zixuan Ma, Tuowei Wang, Shuhong Huang, Xupeng Miao, Shizhi Tang, Kezhao Huang, Zhihao Jia, OSDI, 2023

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Optimal Kernel Orchestration for Tensor Programs
with Korch](https://dl.acm.org/doi/pdf/10.1145/3620666.3651383) - Muyan Hu, Ashwin Venkatram, Shreyashri Biswas, Balamurugan Marimuthu, Bohan Hou, Gabriele Oliaro, Haojie Wang, Liyan Zheng, Xupeng Miao, Jidong Zhai, and Zhihao Jia, ASPLOS, 2024

- <img src="https://img.shields.io/badge/19-pages-green.svg" alt="19-pages" align="left"> [PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections](https://www.usenix.org/system/files/osdi21-wang-haojie.pdf) - Haojie Wang, Jidong Zhai, Mingyu Gao, Zixuan Ma, Shizhi Tang, Liyan Zheng, Yuanzhi Li, Kaiyuan Rong, Yuanyong Chen, and Zhihao Jia, OSDI, 2021

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10071018) - Size Zheng, Siyuan Chen, Peidi Song, Renze Chen, Xiuhong Li, Shengen Yan, Dahua Lin, Jingwen Leng, Yun Liang, HPCA, 2023

Graph Optimization
- <img src="https://img.shields.io/badge/9-pages-green.svg" alt="9-pages" align="left"> [Memory-aware Scheduling for Complex Wired Networks with Iterative Graph Optimization](https://arxiv.org/pdf/2308.13898) - Shuzhang Zhong, Meng Li, Yun Liang, Runsheng Wang, Ru Huang, ICCAD, 2023

- <img src="https://img.shields.io/badge/20-pages-green.svg" alt="20-pages" align="left"> [Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators](https://www.usenix.org/system/files/osdi23-zhao.pdf) - Jie Zhao, Siyuan Feng, Xiaoqiang Dan, Fei Liu, Chengke Wang, Sheng Yuan, Wenyuan Lv, Qikai Xie, OSDI, 2023

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions](https://www.cs.cmu.edu/~zhihaoj2/papers/sosp19.pdf) - Zhihao Jia, Oded Padon, James Thomas, Todd Warszawski, Matei Zaharia, Alex Aiken, SOSP, 2019

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [Optimizing DNN Computation with Relaxed Graph Substitutions](https://www.cs.cmu.edu/~zhihaoj2/papers/sysml19b.pdf) - Zhihao Jia, James Thomas, Todd Warszawski, Mingyu Gao, Matei Zaharia, Alex Aiken, MLSys, 2019


Performance Prediction of Tensor Programs
- <img src="https://img.shields.io/badge/6-pages-green.svg" alt="6-pages" align="left"> [Crop: An Analytical Cost Model for Cross-Platform Performance Prediction of Tensor Programs](https://dl.acm.org/doi/pdf/10.1145/3649329.3658249) - Xinyu Sun, Yu Zhang, Shuo Liu, Yi Zhai, DAC, 2024

Recursive Deep Learning Models
- <img src="https://img.shields.io/badge/17-pages-green.svg" alt="17-pages" align="left"> [Cortex: A compiler for recursive deep learning models](https://proceedings.mlsys.org/paper_files/paper/2021/file/eca986d585a03890a412587a2f5ccb43-Paper.pdf) - Pratik Fegade, Tianqi Chen, Phillip B.Gibbons, Todd C.Mowry, MLSys, 2021

Dynamic Tensor Programs
- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [DietCode: Automatic optimization for dynamic tensor programs](https://proceedings.mlsys.org/paper_files/paper/2022/file/f89b79c9a28d4cae22ef9e557d9fa191-Paper.pdf) - Bojian Zheng, Ziheng Jiang, Cody Hao Yu, Haichen Shen, Josh Fromm, Yizhi Liu, Yida Wang, Luis Ceze, Tianqi Chen, Gennady Pekhimenko, MLSys, 2022

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [SoD²: Statically Optimizing Dynamic Deep Neural Network Execution](https://arxiv.org/pdf/2403.00176) - Wei Niu, Gagan Agrawal, Bin Ren, ASPLOS, 2024

Sparse Tensor Program Optimization
- <img src="https://img.shields.io/badge/30-pages-green.svg" alt="30-pages" align="left"> [A Sparse Iteration Space Transformation Framework for Sparse Tensor Algebra](https://dl.acm.org/doi/pdf/10.1145/3428226) - Ryan Senanayake, Changwan Hong, Ziheng Wang, Amalee Wilson, Stephen Chou, Shaoaib Kamil, Saman Amarasinghe, Fredrik Kjolstad, OOPSLA, 2020

- <img src="https://img.shields.io/badge/30-pages-green.svg" alt="30-pages" align="left"> [Autoscheduling for Sparse Tensor Algebra with an Asymptotic Cost Model](https://dl.acm.org/doi/pdf/10.1145/3519939.3523442) - Willow Ahrens, Fredrik Kjolstad, Saman Amarasinghe, PLDI, 2022


ML Tensor Operations Optimization
- <img src="https://img.shields.io/badge/20-pages-green.svg" alt="20-pages" align="left"> [A Tensor Compiler for Unified Machine Learning Prediction Serving](https://www.usenix.org/system/files/osdi20-nakandala.pdf) - Supun Nakandalac, Karla Saurm, Gyeong-In Yus, Konstantinos Karanasosm, Carlo Curinom, Markus Weimerm, Matteo Interlandim, OSDI, 2020

- <img src="https://img.shields.io/badge/12-pages-green.svg" alt="12-pages" align="left"> [CMLCompiler: A Unified Compiler for Classical Machine Learning](https://dl.acm.org/doi/pdf/10.1145/3577193.3593710) - Xu Wen, Wanling Gao, Anzheng Li, Lei Wang, Zihan Jiang, Jianfeng Zhan, ICS, 2023

Autotune
- <img src="https://img.shields.io/badge/24-pages-green.svg" alt="24-pages" align="left"> [BaCO: A Fast and Portable Bayesian Compiler Optimization Framework](https://dl.acm.org/doi/pdf/10.1145/3623278.3624770) - Erik Hellsten, Artur Souza, Johannes Lenfers, Rubens Lacouture, Olivia Hsu, Adel Ejjeh, Fredrik Kjolstad, Michel Steuwer, Kunle Olukotun, Luigi Nardi, ASPLOS, 2023

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [Bolt: Bridging the gap between auto-tuners and hardware-native performance](https://proceedings.mlsys.org/paper_files/paper/2022/file/1f8053a67ec8e0b57455713cefdd8218-Paper.pdf) - Jiarong Xing, Leyuan Wang, Shang Zhang, Jack Chen, Ang Chen, Yibo Zhu, MLSys, 2022

Automatic
- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [AMOS: Enabling Automatic Mapping for Tensor Computations On Spatial Accelerators with Hardware Abstraction](https://dl.acm.org/doi/pdf/10.1145/3470496.3527440) - Size Zheng, Renze Chen, Anjiang Wei, Yicheng Jin, Qin Han, Liqiang Lu, Bingyang Wu, Xiuhong Li, Shengen Yan, Yun Liang, ISCA, 2022

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations](https://01.me/files/AKG/akg-pldi21.pdf) - Jie Zhao, Bojie Li, Zhen Geng, Renwei Zhang, Xiong Gao, Bin Cheng, Chen Wu, Yun Cheng, Zheng Li, Peng Di, Kun Zhang, Xuefeng Jin, PLDI, 2021

- <img src="https://img.shields.io/badge/18-pages-green.svg" alt="18-pages" align="left"> [Enabling Tensor Language Model to Assist in Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/system/files/osdi24-zhai.pdf) - Yi Zhai, Sijia Yang, Keyu Pan, Renwei Zhang, Shuo Liu, Chao Liu, Zichun Ye, Jianmin Ji, Jie Zhao, Yu Zhang, Yanyong Zhang, OSDI, 2024

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [Bring Your Own Codegen to Deep Learning Compiler](https://arxiv.org/pdf/2105.03215) - Zhi Chen, Cody Hao Yu, Trevor Morris, Jorn Tuyls, Yi-Hsiang Lai, Jared Roesch, Elliott Delaye, Vin Sharma, Yida Wang, arxiv, 2021

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [Hidet: Task-mapping programming paradigm for deep learning tensor programs](https://dl.acm.org/doi/pdf/10.1145/3575693.3575702) - Yaoyao Ding, Cody Hao Yu, Bojian Zheng, Yizhi Liu, Yida Wang, Gennady Pekhimenko, ASPLOS, 2023

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [DISTAL: The Distributed Tensor Algebra Compiler](https://dl.acm.org/doi/pdf/10.1145/3519939.3523437) - Rohan Yadav, Alex Aiken, Fredrik Kjolstad, PLDI, 2022

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [SmartMem: Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile](https://arxiv.org/pdf/2404.13528) - Wei Niu, Md Musfiqur Rahman Sanim, Zhihao Shu, Jiexiong Guan, Xipeng Shen, Miao Yin, Gagan Agrawal, Bin Ren, ASPLOS, 2024

- <img src="https://img.shields.io/badge/18-pages-green.svg" alt="18-pages" align="left"> [GCD²: A Globally Optimizing Compiler for Mapping DNNs to Mobile DSPs](https://par.nsf.gov/servlets/purl/10417473) - Wei Niu, Jiexiong Guan, Xipeng Shen, Yanzhi Wang, Gagan Agrawal, Bin Ren, MICRO, 2022

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [ETO: Accelerating Optimization of DNN Operators by
High-Performance Tensor Program Reuse](https://www.vldb.org/pvldb/vol15/p183-chen.pdf) -Jingzhi Fang, Yanyan Shen, Yue Wan, Lei Chen, VLDB, 2022