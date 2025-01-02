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
- <img src="https://img.shields.io/badge/24-pages-green.svg" alt="24-pages" align="left"> [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/pdf/2002.03794) - Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan, Guangwen Yang, Depei Qian, IEEE Transactions on Parallel and Distributed Systems, 2021 ![Check](https://img.shields.io/badge/✓-done-green)

Dense Tensor Program Optimization
- <img src="https://img.shields.io/badge/6-pages-green.svg" alt="6-pages" align="left"> [A Holistic Functionalization Approach to Optimizing Imperative Tensor Programs in Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3649329.3658483) - Jinming Ma, Xiuhong Li, Zihan Wang Xingcheng Zhang, Shengen Yan, Yuting Chen, Yueqian Zhang, Minxi Jin, Lijuan Jiang, Yun (Eric) Liang, Chao Yang, Dahua Lin, DAC, 2024

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [A Learned Performance Model For Tensor Processing Units](https://proceedings.mlsys.org/paper_files/paper/2021/file/6bcfac823d40046dca25ef6d6d59cc3f-Paper.pdf) - Samuel J.Kaufman, Phitchaya Mangpo Phothilimthana, Yanqi Zhou, Charith Mendis, Sudip Roy, Amit Sabne, Mike Burrows, MLSys, 2021

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2022_2023/papers/ZHENG_ASPLOS_2020.pdf) - Size Zheng, Yun Liang, Shuo Wang, Renze Chen, Kaiwen Sheng, ASPLOS, 2020 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/45-pages-green.svg" alt="45-pages" align="left"> [Modeling the Interplay between Loop Tiling and Fusion in Optimizing Compilers Using Affine Relations](https://dl.acm.org/doi/pdf/10.1145/3635305) - Jie Zhao, Jinchen Xu, Peng Di, Wang Nie, Jiahui Hu, Yanzhi Yi, Sijia Yang, Zhen Geng, Renwei Zhang, Bojie Li, Zhiliang Gan, and Xuefeng Jin, TOCS, 2024

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Optimizing the Memory Hierarchy by Compositing Automatic Transformations on Computations and Data](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9251965) - Jie Zhao, Peng Di, MICRO, 2020

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Tensor Program Optimization with Probabilistic Programs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9251965) - Junru Shao, Xiyou Zhou, Siyuan Feng, Bohan Hou, Ruihang Lai, Hongyi Jin, Wuwei Lin, Masahiro Masuda, Cody Hao Yu, Tianqi Chen, NIPS, 2022 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/43-pages-green.svg" alt="43-pages" align="left"> [Composable and modular code generation in MLIR: A structured and retargetable approach to tensor compiler construction](https://arxiv.org/pdf/2202.03293) - Vasilache Nicolas, Zinenko Oleksandr, Bik Aart JC, Ravishankar Mahesh, Raoux Thomas, Belyaev  Alexander, Springer Matthias, Gysi Tobias, Caballero Diego, Herhut Stephan, Laurenzo Stella, Cohen Albert, arxiv, 2022

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion](https://dl.acm.org/doi/pdf/10.1145/3453483.3454083) - Wei Niu, Jiexiong Guan, Yanzhi Wang, Gagan Agrawal, Bin Ren, PLDI, 2021

- <img src="https://img.shields.io/badge/18-pages-green.svg" alt="18-pages" align="left"> [EINNET: Optimizing Tensor Programs with Derivation-Based Transformations](https://www.usenix.org/system/files/osdi23-zheng.pdf) - Liyan Zheng, Haojie Wang, Jidong Zhai, Muyan Hu, Zixuan Ma, Tuowei Wang, Shuhong Huang, Xupeng Miao, Shizhi Tang, Kezhao Huang, Zhihao Jia, OSDI, 2023 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Optimal Kernel Orchestration for Tensor Programs with Korch](https://dl.acm.org/doi/pdf/10.1145/3620666.3651383) - Muyan Hu, Ashwin Venkatram, Shreyashri Biswas, Balamurugan Marimuthu, Bohan Hou, Gabriele Oliaro, Haojie Wang, Liyan Zheng, Xupeng Miao, Jidong Zhai, and Zhihao Jia, ASPLOS, 2024

- <img src="https://img.shields.io/badge/19-pages-green.svg" alt="19-pages" align="left"> [PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections](https://www.usenix.org/system/files/osdi21-wang-haojie.pdf) - Haojie Wang, Jidong Zhai, Mingyu Gao, Zixuan Ma, Shizhi Tang, Liyan Zheng, Yuanzhi Li, Kaiyuan Rong, Yuanyong Chen, and Zhihao Jia, OSDI, 2021 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10071018) - Size Zheng, Siyuan Chen, Peidi Song, Renze Chen, Xiuhong Li, Shengen Yan, Dahua Lin, Jingwen Leng, Yun Liang, HPCA, 2023

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [UNIT: Unifying tensorized instruction compilation](https://arxiv.org/pdf/2101.08458) - Jian Weng, Animesh Jain, Jie Wang, Leyuan Wang, Yida Wang, Tony Nowatzki, CGO, 2021

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [Tensor Program Optimization with Probabilistic Programs](https://proceedings.neurips.cc/paper_files/paper/2022/file/e894eafae43e68b4c8dfdacf742bcbf3-Paper-Conference.pdf) - Junru Shao, Xiyou Zhou, Siyuan Feng, Bohan Hou, Ruihang Lai, Hongyi Jin, Wuwei Lin, Masahiro Masuda, Cody Hao Yu, Tianqi Chen, NIPS, 2022 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/45-pages-green.svg" alt="45-pages" align="left"> [Modeling the Interplay between Loop Tiling and Fusion in Optimizing Compilers Using Affine Relations](https://dl.acm.org/doi/pdf/10.1145/3635305) - Jie Zhao, Jinchen Xu, Peng Di, Wang Nie, Jiahui Hu, Yanzhi Yi, Sijia Yang, Zhen Geng, Renwei Zhang, Bojie Li, Zhiliang Gan, Xuefeng Jin, TOCS, 2024

- <img src="https://img.shields.io/badge/41-pages-green.svg" alt="41-pages" align="left"> [Relay: A High-Level IR for Deep Learning](https://d1wqtxts1xzle7.cloudfront.net/93907908/1904.08368v1-libre.pdf?1667936889=&response-content-disposition=inline%3B+filename%3DRelay_A_High_Level_IR_for_Deep_Learning.pdf&Expires=1734664090&Signature=hNNPmF5TkPbj99Lf0G4oLGpV0h87pGCppTaWbCC1HxPIUzbytjT-cc6w9zjY1Pl4GU0SDAD8wgX-vouST7CKnSrqo9razwfKblzwOmy6RBje3uqqumq8KdeMn6fUCugFTfgQF05gDcFgWBDAyCMENzhYq0PDuRfHG3qI8BbgSjBiO9FlAXyRNO~SmUXzfs9WSBJhEAljvI7majYepErc0XsPBTPSsaS2BRlqOgEaMXnosz0h7tA2iXjOisDkjXl-77Gn51ir-5ukkxKa0zvcnnMTOfTKdR8daeeT1PaQrCJ5W2zhMebVhcpjcK5yD06QtNBfae~LFr0WJ2da-uMWdg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) - Jared Roesch, Steven Lyubomirsky, Marisa Kirisame, Josh Pollock, Logan Weber, Ziheng Jiang, Tianqi Chen, Thierry Moreau, Zachary Tatlock, arxiv, 2019

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [FreeTensor: A Free-Form DSL with Holistic Optimizations for Irregular Tensor Programs](https://dl.acm.org/doi/pdf/10.1145/3519939.3523448) - Shizhi Tang, Jidong Zhai, Haojie Wang, Lin Jiang, Liyan Zheng, Zhenhao Yuan, Chen Zhang, PLDI, 2022

- <img src="https://img.shields.io/badge/18-pages-green.svg" alt="18-pages" align="left"> [Uncovering Nested Data Parallelism and Data Reuse in DNN Computation with FractalTensor](https://dl.acm.org/doi/pdf/10.1145/3694715.3695961) - Siran Liu, Chengxiang Qi, Ying Cao, Chao Yang, Weifang Hu, Xuanhua Shi, Fan Yang, Mao Yang, SOSP, 2024

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators](https://dl.acm.org/doi/pdf/10.1109/SC41406.2024.00040) - Zheng Zhang, Donglin Yang, Xiaobo Zhou, Dazhao Cheng, SC, 2024

- <img src="https://img.shields.io/badge/12-pages-green.svg" alt="12-pages" align="left"> [Fireiron: A Data-Movement-Aware Scheduling Language for GPUs](https://dl.acm.org/doi/pdf/10.1145/3410463.3414632) - Bastian Hagedorn, Archibald Samuel Elliott, Henrik Barthels, Rastislav Bodik, PACT, 2020


Graph Optimization
- <img src="https://img.shields.io/badge/9-pages-green.svg" alt="9-pages" align="left"> [Memory-aware Scheduling for Complex Wired Networks with Iterative Graph Optimization](https://arxiv.org/pdf/2308.13898) - Shuzhang Zhong, Meng Li, Yun Liang, Runsheng Wang, Ru Huang, ICCAD, 2023

- <img src="https://img.shields.io/badge/20-pages-green.svg" alt="20-pages" align="left"> [Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators](https://www.usenix.org/system/files/osdi23-zhao.pdf) - Jie Zhao, Siyuan Feng, Xiaoqiang Dan, Fei Liu, Chengke Wang, Sheng Yuan, Wenyuan Lv, Qikai Xie, OSDI, 2023

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions](https://www.cs.cmu.edu/~zhihaoj2/papers/sosp19.pdf) - Zhihao Jia, Oded Padon, James Thomas, Todd Warszawski, Matei Zaharia, Alex Aiken, SOSP, 2019

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [Optimizing DNN Computation with Relaxed Graph Substitutions](https://www.cs.cmu.edu/~zhihaoj2/papers/sysml19b.pdf) - Zhihao Jia, James Thomas, Todd Warszawski, Mingyu Gao, Matei Zaharia, Alex Aiken, MLSys, 2019

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Alcop: Automatic load-compute pipelining in deep learning compiler for ai-gpus](https://proceedings.mlsys.org/paper_files/paper/2023/file/d6cde2c1b161daa31be560d062cf2251-Paper-mlsys2023.pdf) - Guyue Huang, Yang Bai, Liu Liu, Yuke Wang, Bei Yu, Yufei Ding, Yuan Xie, MLSys, 2023

- <img src="https://img.shields.io/badge/9-pages-green.svg" alt="9-pages" align="left"> [AutoGraph: Optimizing DNN Computation Graph for Parallel GPU Kernel Execution](https://ojs.aaai.org/index.php/AAAI/article/view/26343) - Yuxuan Zhao, Qi Sun, Zhuolun He, Yang Bai, Bei Yu, AAAI, 2023

- <img src="https://img.shields.io/badge/11-pages-green.svg" alt="11-pages" align="left"> [POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging](https://ojs.aaai.org/index.php/AAAI/article/view/26343) - Shishir G. Patil, Paras Jain, Prabal Dutta, Ion Stoica, Joseph E. Gonzalez, ICML, 2022

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [Collage: Seamless Integration of Deep Learning Backends with Automatic Placement](https://arxiv.org/pdf/2111.00655) - Byungsoo Jeon, Sunghyun Park, Peiyuan Liao, Sheng Xu, Tianqi Chen, Zhihao Jia, PACT, 2022

- <img src="https://img.shields.io/badge/19-pages-green.svg" alt="19-pages" align="left"> [Apollo: Automatic Partition-based Operator Fusion through Layer by Layer Optimization](https://proceedings.mlsys.org/paper_files/paper/2022/file/e175e8a86d28d935be4f43719651f86d-Paper.pdf) - Jie Zhao, Xiong Gao, Ruijie Xia, Zhaochuang Zhang, Deshi Chen, Lei Chen, Renwei Zhang, Zhen Geng, Bin Cheng, and Xuefeng Jin, MLSys, 2022

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [Equality Saturation for Tensor Graph Superoptimization](https://arxiv.org/pdf/2101.01332) - Yichen Yang, Phitchaya Mangpo Phothilimtha, Yisu Remy Wang, Max Willsey, Sudip Roy, Jacques Pienaar, MLSys, 2021

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [IOS: Inter-Operator Scheduler for CNN Acceleration](https://proceedings.mlsys.org/paper_files/paper/2021/file/1f8053a67ec8e0b57455713cefdd8218-Paper.pdf) - Yaoyao Ding, Ligeng Zhu, Zhihao Jia, Gennady Pekhimenko, Song Han, MLSys, 2021

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [Optimizing DNN computation graph using graph substitutions](https://dl.acm.org/doi/pdf/10.14778/3407790.3407857) - Jingzhi Fang, Yanyan Shen, Yue Wang, Lei Chen, VLDB, 2020

- <img src="https://img.shields.io/badge/12-pages-green.svg" alt="12-pages" align="left"> [Transferable Graph Optimizers for ML Compilers](https://proceedings.neurips.cc/paper_files/paper/2020/file/9f29450d2eb58feb555078bdefe28aa5-Paper.pdf) - Yanqi Zhou, Sudip Roy, Amirali Abdolrashidi, Daniel Wong, Peter Ma, Qiumin Xu, Hanxiao Liu, Phitchaya Phothilimtha, Shen Wang, Anna Goldie, Azalia Mirhoseini, James Laudon, NIPS, 2020

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [FusionStitching: Boosting Memory Intensive Computations for Deep Learning Workloads](https://arxiv.org/pdf/2009.10924) - Zhen Zheng, Pengzhan Zhao, Guoping Long, Feiwen Zhu, Kai Zhu, Wenyi Zhao, Lansong Diao, Jun Yang, Wei Lin, arxiv, 2020

- <img src="https://img.shields.io/badge/12-pages-green.svg" alt="12-pages" align="left"> [Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/5f0ad4db43d8723d18169b2e4817a160-Paper.pdf) - Woosuk Kwon, Gyeong-In Yu, Eunji Jeong, Byung-Gon Chun, NIPS, 2020

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [Stateful dataflow multigraphs: a data-centric model for performance portability on heterogeneous architectures](https://dl.acm.org/doi/pdf/10.1145/3295500.3356173) - Tal Ben-Nun, Johannes de Fine Licht, Alexandros N. Ziogas, Timo Schneider, Torsten Hoefler, SC, 2019


Performance Prediction of Tensor Programs
- <img src="https://img.shields.io/badge/6-pages-green.svg" alt="6-pages" align="left"> [Crop: An Analytical Cost Model for Cross-Platform Performance Prediction of Tensor Programs](https://dl.acm.org/doi/pdf/10.1145/3649329.3658249) - Xinyu Sun, Yu Zhang, Shuo Liu, Yi Zhai, DAC, 2024

- <img src="https://img.shields.io/badge/7-pages-green.svg" alt="7-pages" align="left"> [Effective Performance Modeling and Domain-Specific Compiler Optimization of CNNs for GPUs](https://dl.acm.org/doi/pdf/10.1145/3559009.3569674) - Zhihe Zhao, Xian Shuai, Yang Bai, Neiwen Ling, Nan Guan, Zhenyu Yan, Guoliang Xing, arxiv, 2022

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [Moses: Efficient exploitation of cross-device transferable features for tensor program optimization](https://arxiv.org/pdf/2201.05752) - Yufan Xu, Qiwei Yuan, Erik Curtis Barton, Rui Li, P. Sadayappan, Aravind Sukumaran-Rajam, PACT, 2022


Recursive Deep Learning Models
- <img src="https://img.shields.io/badge/17-pages-green.svg" alt="17-pages" align="left"> [Cortex: A compiler for recursive deep learning models](https://proceedings.mlsys.org/paper_files/paper/2021/file/eca986d585a03890a412587a2f5ccb43-Paper.pdf) - Pratik Fegade, Tianqi Chen, Phillip B.Gibbons, Todd C.Mowry, MLSys, 2021

- <img src="https://img.shields.io/badge/19-pages-green.svg" alt="19-pages" align="left"> [RECom: A Compiler Approach to Accelerate Recommendation Model Inference with Massive Embedding Columns](https://jamesthez.github.io/files/recom-asplos23.pdf) - Zaifeng Pan, Zhen Zheng, Feng Zhang, Ruofan Wu, Hao Liang, Dalin Wang, Xiafei Qiu, Junjie Bai, Wei Lin, Xiaoyong Du, ASPLOS, 2023

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [RecFlex: Enabling Feature Heterogeneity-Aware Optimization for Deep Recommendation Models with Flexible Schedules](https://panzaifeng.github.io/assets/pdf/sc24recflex.pdf) - Zaifeng Pan, Zhen Zheng, Feng Zhang, Bing Xie, Ruofan Wu, Shaden Smith, Chuanjie Liu, Olatunji Ruwase, Xiaoyong Du, Yufei Ding, SC, 2024

GNN Optimization
- <img src="https://img.shields.io/badge/17-pages-green.svg" alt="17-pages" align="left"> [WiseGraph: Optimizing GNN with Joint Workload Partition of Graph and Operations](https://jamesthez.github.io/files/wisegraph-eurosys24.pdf) - Kezhao Huang, Jidong Zhai, Liyan Zheng, Haojie Wang, Yuyang Jin, Qihao Zhang, Runqing Zhang, Zhen Zheng, Youngmin Yi, Xipeng Shen, Eurosys, 2024

- <img src="https://img.shields.io/badge/17-pages-green.svg" alt="17-pages" align="left"> [Hector: An Efficient Programming and Compilation Framework for Implementing Relational Graph Neural Networks in GPU Architectures](https://dl.acm.org/doi/pdf/10.1145/3620666.3651322) - Kun Wu, Mert Hidayetoğlu, Xiang Song, Sitao Huang, Da Zheng, Israt Nisa, Wen-Mei Hwu, ASPLOS, 2024

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [Graphiler: Optimizing Graph Neural Networks with Message Passing Data Flow Graph](https://proceedings.mlsys.org/paper_files/paper/2022/file/a1126573153ad7e9f44ba80e99316482-Paper.pdf) - Zhiqiang Xie, Minjie Wang, Zihao Ye, Zheng Zhang, Rui Fan, MLSys, 2022

- <img src="https://img.shields.io/badge/17-pages-green.svg" alt="17-pages" align="left"> [Seastar: vertex-centric programming for graph neural networks](https://dl.acm.org/doi/pdf/10.1145/3447786.3456247) - Yidi Wu, Kaihao Ma, Zhenkun Cai, Tatiana Jin, Boyang Li, Chenguang Zheng, James Cheng, Fan Yu, EuroSys, 2021

- <img src="https://img.shields.io/badge/12-pages-green.svg" alt="12-pages" align="left"> [FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems](https://arxiv.org/pdf/2008.11359) - Yuwei Hu, Zihao Ye, Minjie Wang, Jiali Yu, Da Zheng, Mu Li, Zheng Zhang, Zhiru Zhang, Yida Wang, SC, 2020

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [AutoTransfer: AutoML with Knowledge Transfer - An Application to Graph Neural Networks ](https://openreview.net/pdf?id=y81ppNf_vg) - Kaidi Cao, Jiaxuan You, Jiaju Liu, Jure Leskovec, ICLR, 2023


Dynamic Tensor Programs
- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [DietCode: Automatic optimization for dynamic tensor programs](https://proceedings.mlsys.org/paper_files/paper/2022/file/f89b79c9a28d4cae22ef9e557d9fa191-Paper.pdf) - Bojian Zheng, Ziheng Jiang, Cody Hao Yu, Haichen Shen, Josh Fromm, Yizhi Liu, Yida Wang, Luis Ceze, Tianqi Chen, Gennady Pekhimenko, MLSys, 2022

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [SoD²: Statically Optimizing Dynamic Deep Neural Network Execution](https://arxiv.org/pdf/2403.00176) - Wei Niu, Gagan Agrawal, Bin Ren, ASPLOS, 2024

- <img src="https://img.shields.io/badge/29-pages-green.svg" alt="29-pages" align="left"> [BladeDISC: Optimizing Dynamic Shape Machine Learning Workloads via Compiler Approach](https://jamesthez.github.io/files/bladedisc-sigmod24.pdf) - Zhen Zheng, Zaifeng Pan, Dalin Wang, Kai Zhu, Wenyi Zhao, Tianyou Guo, Xiafei Qiu, Minmin Sun, Junjie Bai, Feng Zhang, Xiaoyong Du, Jidong Zhai, Wei Lin, SIGMOD, 2024

- <img src="https://img.shields.io/badge/7-pages-green.svg" alt="7-pages" align="left"> [DISC : A Dynamic Shape Compiler for Machine Learning Workloads](https://arxiv.org/pdf/2103.05288) - Kai Zhu, Wenyi Zhao, Zhen Zheng, Tianyou Guo, Pengzhan Zhao, Feiwen Zhu, Junjie Bai, Jun Yang, Xiaoyong Liu, Lansong Diao, Wei Lin, MLSys, 2021

- <img src="https://img.shields.io/badge/20-pages-green.svg" alt="20-pages" align="left"> [Optimizing Dynamic Neural Networks with Brainstorm](https://www.usenix.org/system/files/osdi23-cui.pdf) - Weihao Cui, Zhenhua Han, Lingji Ouyang, Yichuan Wang, Ningxin Zheng, Lingxiao Ma, Yuqing Yang, Fan Yang, Jilong Xue, Lili Qiu, Lidong Zhou, Quan Chen, Haisheng Tan, Minyi Guo, OSDI, 2023

- <img src="https://img.shields.io/badge/17-pages-green.svg" alt="17-pages" align="left"> [Grape: Practical and Efficient Graphed Execution for Dynamic Deep Neural Networks on GPUs](https://dl.acm.org/doi/pdf/10.1145/3620666.3651322) - Bojian Zheng, Cody Hao Yu, Jie Wang, Yaoyao Ding, Yizhi Liu, Yida Wang, Gennady Pekhimenko, ASPLOS, 2024

- <img src="https://img.shields.io/badge/17-pages-green.svg" alt="17-pages" align="left"> [Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization](https://proceedings.mlsys.org/paper_files/paper/2021/file/5b47430e24a5a1f9fe21f0e8eb814131-Paper.pdf) - Kun Wu, Mert Hidayetoğlu, Xiang Song, Sitao Huang, Da Zheng, Israt Nisa, Wen-Mei Hwu, MICRO, 2023

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Nimble: Efficiently Compiling Dynamic Neural Networks for Model Inference](https://dl.acm.org/doi/pdf/10.1145/3613424.3614248) - Haichen Shen, Jared Roesch, Zhi Chen, Wei Chen, Yong Wu, Mu Li, Vin Sharma, Zachary Tatlock, Yida Wang, MLSys, 2021

- <img src="https://img.shields.io/badge/27-pages-green.svg" alt="27-pages" align="left"> [The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding](https://proceedings.mlsys.org/paper_files/paper/2022/file/afe8a4577080504b8bec07bbe4b2b9cc-Paper.pdf) - Pratik Fegade, Tianqi Chen, Phillip Gibbons, Todd Mowry, MLSys, 2022

- <img src="https://img.shields.io/badge/27-pages-green.svg" alt="27-pages" align="left"> [Axon: A Language for Dynamic Shapes in Deep Learning Graphs](https://arxiv.org/pdf/2210.02374) - Alexander Collins, Vinod Grover, arxiv, 2022

Sparse Tensor Program Optimization
- <img src="https://img.shields.io/badge/30-pages-green.svg" alt="30-pages" align="left"> [A Sparse Iteration Space Transformation Framework for Sparse Tensor Algebra](https://dl.acm.org/doi/pdf/10.1145/3428226) - Ryan Senanayake, Changwan Hong, Ziheng Wang, Amalee Wilson, Stephen Chou, Shaoaib Kamil, Saman Amarasinghe, Fredrik Kjolstad, OOPSLA, 2020

- <img src="https://img.shields.io/badge/30-pages-green.svg" alt="30-pages" align="left"> [Autoscheduling for Sparse Tensor Algebra with an Asymptotic Cost Model](https://dl.acm.org/doi/pdf/10.1145/3519939.3523442) - Willow Ahrens, Fredrik Kjolstad, Saman Amarasinghe, PLDI, 2022


ML Tensor Operations Optimization
- <img src="https://img.shields.io/badge/20-pages-green.svg" alt="20-pages" align="left"> [A Tensor Compiler for Unified Machine Learning Prediction Serving](https://www.usenix.org/system/files/osdi20-nakandala.pdf) - Supun Nakandalac, Karla Saurm, Gyeong-In Yus, Konstantinos Karanasosm, Carlo Curinom, Markus Weimerm, Matteo Interlandim, OSDI, 2020 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/12-pages-green.svg" alt="12-pages" align="left"> [CMLCompiler: A Unified Compiler for Classical Machine Learning](https://dl.acm.org/doi/pdf/10.1145/3577193.3593710) - Xu Wen, Wanling Gao, Anzheng Li, Lei Wang, Zihan Jiang, Jianfeng Zhan, ICS, 2023 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/17-pages-green.svg" alt="17-pages" align="left"> [SilvanForge: A Schedule Guided Retargetable Compiler for Decision Tree Inference](https://dl.acm.org/doi/pdf/10.1145/3694715.3695958) - Ashwin Prasad, Sampath Rajendra, Kaushik Rajan, R Govindarajan, Uday Bondhugula, SOSP, 2024 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/18-pages-green.svg" alt="18-pages" align="left"> [Treebeard: An Optimizing Compiler for Decision Tree Based ML Inference](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9923840) - Ashwin Prasad, Sampath Rajendra, Kaushik Rajan, R Govindarajan, Uday Bondhugula, MICRO, 2022 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/12-pages-green.svg" alt="12-pages" align="left"> [Tile Size and Loop Order Selection using Machine Learning for Multi-/Many-Core Architectures](https://dl.acm.org/doi/pdf/10.1145/3650200.3656630) - Shilpa Babalad, Shirish Shevade, Matthew Jacob Thazhuthaveetil, R Govindarajan, ICS, 2024 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [A Flexible Approach to Autotuning Multi-Pass Machine Learning Compilers](https://mangpo.net/papers/xla-autotuning-pact2021.pdf) - Phitchaya Mangpo Phothilimthana, Amit Sabne, Nikhil Sarda, Karthik Srinivasa Murthy, Yanqi Zhou, Christof Angermueller, Mike Burrows, Sudip Roy, Ketan Mandke, Rezsa Farahani, Yu Emma Wang, Berkin Ilbeyi, Blake Hechtman, Bjarke Roune, Shen Wang, Yuanzhong Xu, and Samuel J. Kaufman, PACT, 2024

- <img src="https://img.shields.io/badge/25-pages-green.svg" alt="25-pages" align="left"> [TreeHouse: An MLIR-based Compilation Flow for Real-Time Tree-based Inference](https://dl.acm.org/doi/pdf/10.1145/3704727) - ChiaHui Su, Chia-Hua Ku, Jenq Kuen Lee, Kuan-Hsun Chen, TECS, 2024

- <img src="https://img.shields.io/badge/26-pages-green.svg" alt="26-pages" align="left"> [Efficient Realization of Decision Trees for Real-Time Inference](https://dl.acm.org/doi/pdf/10.1145/3508019) - Kuan-Hsun Chen, Chiahui Su, Christian Hakert, Sebastian Buschjäger, Chao-Lin Lee, Jenq-Kuen Lee, Katharina Morik, Jian-Jia Chen, TECS, 2022

Autotune
- <img src="https://img.shields.io/badge/24-pages-green.svg" alt="24-pages" align="left"> [BaCO: A Fast and Portable Bayesian Compiler Optimization Framework](https://dl.acm.org/doi/pdf/10.1145/3623278.3624770) - Erik Hellsten, Artur Souza, Johannes Lenfers, Rubens Lacouture, Olivia Hsu, Adel Ejjeh, Fredrik Kjolstad, Michel Steuwer, Kunle Olukotun, Luigi Nardi, ASPLOS, 2023 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [Bolt: Bridging the gap between auto-tuners and hardware-native performance](https://proceedings.mlsys.org/paper_files/paper/2022/file/1f8053a67ec8e0b57455713cefdd8218-Paper.pdf) - Jiarong Xing, Leyuan Wang, Shang Zhang, Jack Chen, Ang Chen, Yibo Zhu, MLSys, 2022

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [Soter: Analytical Tensor-Architecture Modeling and Automatic Tensor Program Tuning for Spatial Accelerators](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10609702) - Fuyu Wang, Minghua Shen, Yufei Ding, Nong Xiao, ISCA, 2024 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Felix: Optimizing Tensor Programs with Gradient Descent](https://dl.acm.org/doi/pdf/10.1145/3620666.3651348) - Yifan Zhao, Hashim Sharif, Vikram Adve, Sasa Misailovic, ASPLOS, 2024 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Tlp: A deep learning-based cost model for tensor program tuning](https://arxiv.org/pdf/2211.03578) - Yi Zhai, Yu Zhang, Shuo Liu, Xiaomeng Chu, Jie Peng, Jianmin Ji, Yanyong Zhang, ASPLOS, 2023

- <img src="https://img.shields.io/badge/18-pages-green.svg" alt="18-pages" align="left"> [RAMMER: Enabling Holistic Deep Learning Compiler Optimizations with rTasks](https://www.usenix.org/system/files/osdi20-ma.pdf) - Lingxiao Ma, Zhiqiang Xie, Zhi Yang, Jilong Xue, Youshan Miao, Wei Cui, Wenxiang Hu, Fan Yang, Lintao Zhang, Lidong Zhou, OSDI, 2020

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Lorien: Efficient Deep Learning Workloads Delivery](https://assets.amazon.science/c2/46/2481c9064a8bbaebcf389dd5ad75/lorien-efficient-deep-learning-workloads-delivery.pdf) - Cody Hao Yu, Xingjian Shi, Haichen Shen, Zhi Chen, Mu Li, Yida Wang, SOCC, 2021

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [A Practical Tile Size Selection Model for Affine Loop Nests](https://dl.acm.org/doi/pdf/10.1145/3447818.3462213) - Kumudha Narasimhan, Aravind Acharya, Abhinav Baid, Uday Bondhugula, ICS, 2021

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [ATFormer: A Learned Performance Model with Transfer Learning
Across Devices for Deep Learning Tensor Programs](https://aclanthology.org/2023.emnlp-main.250.pdf) - Yang Bai, Wenqian Zhao, Shuo Yin, Zixiao Wang, Bei Yu, EMNLP, 2023

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [AutoGTCO: Graph and tensor co-Optimize for image recognition with transformers on GPU](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9643487) - Yang Bai, Xufeng Yao, Qi Sun, Bei Yu, ICCAD, 2021

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [Neural Architecture Search as Program Transformation Exploration](https://dl.acm.org/doi/pdf/10.1145/3445814.3446753) - Jack Turner, Elliot J. Crowley, Michael F. P. O’Boyle, ASPLOS, 2021 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [One-Shot Tuner for Deep Learning Compilers](https://dl.acm.org/doi/pdf/10.1145/3497776.3517774) - Jaehun Ryu, Eunhyeok Park, Hyojin Sung, CC, 2022

- <img src="https://img.shields.io/badge/27-pages-green.svg" alt="27-pages" align="left"> [PolyDL: Polyhedral Optimizations for Creation of High-performance DL Primitives](https://dl.acm.org/doi/pdf/10.1145/3433103) - Sanket Tavarageri, Alexander Heinecke, Sasikanth Avancha, Bharat Kaul, Gagandeep Goyal, Ramakrishna Upadrasta, TACO, 2021

- <img src="https://img.shields.io/badge/11-pages-green.svg" alt="11-pages" align="left"> [Reinforcement Learning and Adaptive Sampling for Optimized DNN Compilation](https://arxiv.org/pdf/1905.12799) - Byung Hoon Ahn, Prannoy Pilligundla, Hadi Esmaeilzadeh, arxiv, 2019

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [Accelerated Auto-Tuning of GPU Kernels for Tensor Computations](https://dl.acm.org/doi/pdf/10.1145/3650200.3656626) - Chendi Li, Yufan Xu, Sina Mahdipour Saravani, Ponnuswamy Sadayappan, ICS, 2024

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [A full-stack search technique for domain optimized deep learning accelerators](https://dl.acm.org/doi/pdf/10.1145/3650200.3656626) - Dan Zhang, Safeen Huda, Ebrahim Songhori, Kartik Prabhu, Quoc Le, Anna Goldie, Azalia Mirhoseini, ASPLOS, 2022

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [A Flexible Approach to Autotuning Multi-Pass Machine Learning Compilers](https://dl.acm.org/doi/pdf/10.1145/3650200.3656626) - Phitchaya Mangpo Phothilimthana, Amit Sabne, Nikhil Sarda, Karthik Srinivasa Murthy,Yanqi Zhou, Christof Angermueller, PACT, 2021

- <img src="https://img.shields.io/badge/12-pages-green.svg" alt="12-pages" align="left"> [Learning to Optimize Halide with Tree Search and Random Programs](https://dl.acm.org/doi/pdf/10.1145/3650200.3656626) - Andrew Adams, Karima Ma, Luke Anderson, Riyadh Baghdadi, Tzu-Mao Li, Michaël Gharbi, Benoit Steiner, Steven Johnson, Kayvon Fatahalian, Frédo Durand, , TOG, 2019 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [Accelerated Auto-Tuning of GPU Kernels for Tensor Computations](https://dl.acm.org/doi/pdf/10.1145/3650200.3656626) - Chendi Li, Yufan Xu, Sina Mahdipour Saravani, Ponnuswamy Sadayappan, ICS, 2024

- <img src="https://img.shields.io/badge/18-pages-green.svg" alt="18-pages" align="left"> [MonoNN: Enabling a New Monolithic Optimization Space for Neural Network Inference Tasks on Modern GPU-Centric Architectures](https://jamesthez.github.io/files/mononn-osdi24.pdf) - Donglin Zhuang, Zhen Zheng, Haojun Xia, Xiafei Qiu, Junjie Bai, Wei Lin, OSDI, 2024


Automatic
- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [AMOS: Enabling Automatic Mapping for Tensor Computations On Spatial Accelerators with Hardware Abstraction](https://dl.acm.org/doi/pdf/10.1145/3470496.3527440) - Size Zheng, Renze Chen, Anjiang Wei, Yicheng Jin, Qin Han, Liqiang Lu, Bingyang Wu, Xiuhong Li, Shengen Yan, Yun Liang, ISCA, 2022 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations](https://01.me/files/AKG/akg-pldi21.pdf) - Jie Zhao, Bojie Li, Zhen Geng, Renwei Zhang, Xiong Gao, Bin Cheng, Chen Wu, Yun Cheng, Zheng Li, Peng Di, Kun Zhang, Xuefeng Jin, PLDI, 2021

- <img src="https://img.shields.io/badge/18-pages-green.svg" alt="18-pages" align="left"> [Enabling Tensor Language Model to Assist in Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/system/files/osdi24-zhai.pdf) - Yi Zhai, Sijia Yang, Keyu Pan, Renwei Zhang, Shuo Liu, Chao Liu, Zichun Ye, Jianmin Ji, Jie Zhao, Yu Zhang, Yanyong Zhang, OSDI, 2024 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [Bring Your Own Codegen to Deep Learning Compiler](https://arxiv.org/pdf/2105.03215) - Zhi Chen, Cody Hao Yu, Trevor Morris, Jorn Tuyls, Yi-Hsiang Lai, Jared Roesch, Elliott Delaye, Vin Sharma, Yida Wang, arxiv, 2021

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [Hidet: Task-mapping programming paradigm for deep learning tensor programs](https://dl.acm.org/doi/pdf/10.1145/3575693.3575702) - Yaoyao Ding, Cody Hao Yu, Bojian Zheng, Yizhi Liu, Yida Wang, Gennady Pekhimenko, ASPLOS, 2023 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [DISTAL: The Distributed Tensor Algebra Compiler](https://dl.acm.org/doi/pdf/10.1145/3519939.3523437) - Rohan Yadav, Alex Aiken, Fredrik Kjolstad, PLDI, 2022

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [SmartMem: Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile](https://arxiv.org/pdf/2404.13528) - Wei Niu, Md Musfiqur Rahman Sanim, Zhihao Shu, Jiexiong Guan, Xipeng Shen, Miao Yin, Gagan Agrawal, Bin Ren, ASPLOS, 2024

- <img src="https://img.shields.io/badge/18-pages-green.svg" alt="18-pages" align="left"> [GCD²: A Globally Optimizing Compiler for Mapping DNNs to Mobile DSPs](https://par.nsf.gov/servlets/purl/10417473) - Wei Niu, Jiexiong Guan, Xipeng Shen, Yanzhi Wang, Gagan Agrawal, Bin Ren, MICRO, 2022

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [ETO: Accelerating Optimization of DNN Operators by High-Performance Tensor Program Reuse](https://www.vldb.org/pvldb/vol15/p183-chen.pdf) -Jingzhi Fang, Yanyan Shen, Yue Wan, Lei Chen, VLDB, 2022


- <img src="https://img.shields.io/badge/12-pages-green.svg" alt="12-pages" align="left"> [Glow: Graph Lowering Compiler Techniques for Neural Networks](https://arxiv.org/pdf/1805.00907) -Nadav Rotem, Jordan Fix, Saleem Abdulrasool, Garret Catron, Summer Deng, Roman Dzhabarov, Nick Gibson, James Hegeman, Meghan Lele, Roman Levenstein, Jack Montgomery, Bert Maher, Satish Nadathur, Jakob Olesen, Jongsoo Park, Artem Rakhov, Misha Smelyanskiy, Man Wang, arxiv, 2019

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Heron: Automatically constrained high-performance library generation for deep learning accelerators](https://dl.acm.org/doi/pdf/10.1145/3582016.3582061) -Jun Bi, Qi Guo, Xiaqing Li, Yongwei Zhao, Yuanbo Wen,  Yuxuan Guo, Enshuai Zhou, Xing Hu, Zidong Du, Ling Li, Huaping Chen, Tianshi Chen, ASPLOS, 2023

- <del><img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [Hydride: A Retargetable and Extensible Synthesis-based Compiler for Modern Hardware Architectures](https://dl.acm.org/doi/pdf/10.1145/3620665.3640385) -Akash Kothari, Abdul Rafae Noor, Muchen Xu, Hassam Uddin, Dhruv Baronia, Stefanos Baziotis, Vikram Adve, Charith Mendis,  Sudipta Sengupta, ASPLOS, 2024</del>

- <img src="https://img.shields.io/badge/3-pages-green.svg" alt="3-pages" align="left"> [Intel ngraph: An intermediate representation, compiler, and executor for deep learning](https://arxiv.org/pdf/1801.08058) -Scott Cyphers, Arjun K. Bansal, Anahita Bhiwandiwalla, Jayaram Bobba, Matthew Brookhart, Avijit Chakraborty, Will Constable, Christian Convey, Leona Cook, Omar Kanawi, Robert Kimball, Jason Knight, Nikolay Korovaiko, Varun Kumar, Yixing Lao, Christopher R. Lishka, Jaikrishnan Menon, Jennifer Myers, Sandeep Aswath Narayana, Adam Procter, Tristan J. Webb, arxiv, 2018

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [TIRAMISU: A Polyhedral Compiler for Expressing Fast and Portable Code](https://arxiv.org/pdf/1804.10694) -Riyadh Baghdadi, Jessica Ray, Malek Ben Romdhane, Emanuele Del Sozzo, Abdurrahman Akkas, Yunming Zhang, Patricia Suriana, Shoaib Kamil, Saman Amarasinghe, CGO, 2019

- <img src="https://img.shields.io/badge/14-pages-green.svg" alt="14-pages" align="left"> [TensorIR: An abstraction for automatic tensorized program optimization](https://dl.acm.org/doi/pdf/10.1145/3575693.3576933) -Siyuan Feng, Bohan Hou, Hongyi Jin, Wuwei Lin, Junru Shao, Ruihang Lai, Zihao Ye, Lianmin Zheng, Cody Hao Yu, Yong Yu, Tianqi Chen, ASPLOS, 2023 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/37-pages-green.svg" alt="37-pages" align="left"> [Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://dl.acm.org/doi/pdf/10.1145/3575693.3576933) - Nicolas Vasilache, Oleksandr Zinenko, Theodoros Theodoridis, Priya Goyal, Zachary DeVito, William S. Moses, Sven Verdoolaege, Andrew Adams, Albert Cohen, arxiv, 2018

- <img src="https://img.shields.io/badge/13-pages-green.svg" alt="13-pages" align="left"> [MLIR: Scaling Compiler Infrastructure for Domain Specific Computation](https://rcs.uwaterloo.ca/~ali/cs842-s23/papers/mlir.pdf) - Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, Oleksandr Zinenko, CGO, 2021 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/12-pages-green.svg" alt="12-pages" align="left"> [AI Powered Compiler Techniques for DL Code Optimization](https://arxiv.org/pdf/2104.05573) - Sanket Tavarageri, Gagandeep Goyal, Sasikanth Avancha, Bharat Kaul, Ramakrishna Upadrasta, arxiv, 2021

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-intensive ML Training and Inference on Modern SIMT Architectures](https://jamesthez.github.io/files/astitch-asplos22.pdf) - Zhen Zheng, Xuanda Yang, Pengzhan Zhao, Guoping Long, Kai Zhu, Feiwen Zhu, Wenyi Zhao, Xiaoyong Liu, Jun Yang, Jidong Zhai, Shuaiwen Leon Song, Wei Lin, ASPLOS, 2022 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/17-pages-green.svg" alt="17-pages" align="left"> [ROLLER: Fast and Efficient Tensor Compilation for Deep Learning](https://www.usenix.org/system/files/osdi22-zhu.pdf) - Hongyu Zhu, Ruofan Wu, Yijia Diao, Shanbin Ke, Haoyu Li, Chen Zhang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Wei Cui, Fan Yang, Mao Yang, Lidong Zhou, Asaf Cidon, Gennady Pekhimenko, OSDI, 2022

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [Optimizing Deep Learning Inference via Global Analysis and Tensor Expressions](https://dl.acm.org/doi/pdf/10.1145/3617232.3624858) - Chunwei Xia, Jiacheng Zhao, Qianqi Sun, Zheng Wang, Yuan Wen, Teng Yu, Xiaobing Feng, Huimin Cui, ASPLOS, 2024

- <img src="https://img.shields.io/badge/16-pages-green.svg" alt="16-pages" align="left"> [Mind mappings: enabling efficient algorithm-accelerator mapping space search](https://dl.acm.org/doi/pdf/10.1145/3445814.3446762) - Kartik Hegde, Po-An Tsai, Sitao Huang, Vikas Chandra, Angshuman Parashar, Christopher W. Fletcher, ASPLOS, 2021

- <img src="https://img.shields.io/badge/9-pages-green.svg" alt="9-pages" align="left"> [Gamma: Automating the hw mapping of dnn models on accelerators via genetic algorithm](https://dl.acm.org/doi/pdf/10.1145/3400302.3415639) - Sheng-Chun Kao, Tushar Krishna, ICCAD, 2020

- <img src="https://img.shields.io/badge/27-pages-green.svg" alt="27-pages" align="left"> [dMazeRunner: Executing Perfectly Nested Loops on Dataflow Accelerators](https://dl.acm.org/doi/pdf/10.1145/3358198) - Shail Dave, Youngbin Kim, Sasikanth Avancha, Kyoungwoo Lee, Aviral Shrivastava, TECS, 2019

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Analytical Characterization and Design Space Exploration for Optimization of CNNs](https://dl.acm.org/doi/pdf/10.1145/3445814.3446759) - Rui Li, Yufan Xu, Aravind Sukumaran-Rajam, Atanas Rountev, P. Sadayappan, ASPLOS, 2021

- <img src="https://img.shields.io/badge/15-pages-green.svg" alt="15-pages" align="left"> [Interstellar: Using Halide’s Scheduling Language to Analyze DNN Accelerators](https://dl.acm.org/doi/pdf/10.1145/3373376.3378514) - Xuan Yang, Mingyu Gao, Qiaoyi Liu, Jeff Setter, Jing Pu, Ankita Nayak, Steven Bell, Kaidi Cao, Heonjae Ha, Priyanka Raina, Christos Kozyrakis, Mark Horowitz, ASPLOS, 2020 ![Check](https://img.shields.io/badge/✓-done-green)

- <img src="https://img.shields.io/badge/17-pages-green.svg" alt="17-pages" align="left"> [Mind the Gap: Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10609642) - Qijing Huang, Po-An Tsai, Joel S. Emer, Angshuman Parashar, ISCA, 2024


Dataset
- <img src="https://img.shields.io/badge/21-pages-green.svg" alt="21-pages" align="left"> [TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs](https://arxiv.org/pdf/2308.13490) - Phitchaya Mangpo Phothilimthana, Sami Abu-El-Haija, Kaidi Cao, Bahare Fatemi, Charith Mendis, Bryan Perozzi, NIPS, 2023
