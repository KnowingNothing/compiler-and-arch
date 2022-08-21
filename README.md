## Compilers and Architectures
![GitHub last commit](https://img.shields.io/github/last-commit/KnowingNothing/compiler-and-arch)
[![](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A list of awesome compilers for different architectures and emerging domains.

## Contents

- [List of Conferences and Journals Considered](#list-of-conferences-and-journals-considered)
- [Compiler Toolchain](#compiler-toolchain)
- [Compilers for AI chips](#compilers-for-ai-chips)
- [Compilers for PIM](#compilers-for-pim)
- [Compilers for Brain-inspired Hardware](#compilers-for-brain-inspired-hardware)
- [Compilers for SIMT GPU](#compilers-for-simt-gpu)
- [Compilers for CPU](#compilers-for-cpu)
- [Compilers for Mobile and Edge](#compilers-for-mobile-and-edge)
- [Compilers for RISC-V](#compilers-for-risc-v)
- [Compilers for Configurable Hardware](#compilers-for-configurable-hardware)
- [Survey and Books](#survey-and-books)
- [Talks, Tutorials, and Videos](#talks-tutorials-and-videos)


## List of Conferences and Journals Considered
- Conferences
    - ASPLOS, ISCA, MICRO, HPCA
    - OSDI, SOSP, PLDI, PPoPP, SC
    - DAC, ICLR, NeurIPs, ATC, OOPSLA
    - CGO, MLSys, SIGGRAPH, PACT, POPL, ICS
- Journals
    - TPDS, TCAD, TC
- Preprint
    - arXiv


## Compiler Toolchain

- Open-source

- Close-source (binary available)


## Compilers for AI chips

- Auto-tensorization and Auto-vectorization
    - [TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://arxiv.org/abs/2207.04296). **arXiv 2022**. Siyuan Feng, Bohan Hou, Hongyi Jin, Wuwei Lin, Junru Shao, Ruihang Lai, Zihao Ye, Lianmin Zheng, Cody Hao Yu, Yong Yu, Tianqi Chen. _Shanghai Jiao Tong University_.
    - [AMOS: enabling automatic mapping for tensor computations on spatial accelerators with hardware abstraction](https://dl.acm.org/doi/abs/10.1145/3470496.3527440) **ISCA 2022**. [code](https://github.com/KnowingNothing/AMOS). Size Zheng, Renze Chen, Anjiang Wei, Yicheng Jin, Qin Han, Liqiang Lu, Bingyang Wu, Xiuhong Li, Shengen Yan, Yun Liang. _Peking University_.
    - [UNIT: Unifying Tensorized Instruction Compilation](https://polyarch.cs.ucla.edu/papers/cgo2021-unit.pdf) **CGO 2021**. [code](https://github.com/were/UNIT). Jian Weng, Animesh Jain, Jie Wang, Leyuan Wang, Yida Wang, Tony Nowatzki. _University of California, Los Angeles_.

- Polyhedral Optimization
    - [AKG: automatic kernel generation for neural processing units using polyhedral transformations](https://dl.acm.org/doi/abs/10.1145/3453483.3454106) **PLDI 2021**. [code](https://github.com/mindspore-ai/akg). Jie Zhao, Bojie Li, Wang Nie, Zhen Geng, Renwei Zhang, Xiong Gao, Bin Cheng, Chen Wu, Yun Cheng, Zheng Li, Peng Di, Kun Zhang, Xuefeng Jin. _State Key Laboratory of Mathematical Engineering and Advanced Computing, China_.
    - [Optimizing the Memory Hierarchy by Compositing Automatic Transformations on Computations and Data](https://www.microarch.org/micro53/papers/738300a427.pdf) **MICRO 2020**. [code](https://github.com/mindspore-ai/akg). Jie Zhao, Peng Di. _State Key Laboratory of Mathematical Engineering and Advanced Computing, China_.

## Compilers for PIM


## Compilers for Brain-inspired Hardware


## Compilers for SIMT GPU

- Efficient Compute-intensive Kernel Generation

- Efficient Compute-intensive Kernel Fusion

- Efficient Memory-intensive Kernel Fusion

- Polyhedral-based Optimization

- Sparse GPU Kernel Optimization

- Compilers for HPC Workloads on GPU

- Holistic Graph Optimization

- Distributed Optimization


## Compilers for CPU


## Compilers for Mobile and Edge


## Compilers for RISC-V


## Cross-architecture Optimization

- Auto-scheduling and Auto-tuning
    - [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/system/files/osdi20-zheng.pdf) **OSDI 2020**. [code](https://github.com/apache/tvm/tree/main/python/tvm/auto_scheduler). Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, Ion Stoica. _UC Berkeley_.
    - [FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System](https://dl.acm.org/doi/10.1145/3373376.3378508). **ASPLOS 2020**. [code](https://github.com/KnowingNothing/FlexTensor). Size Zheng, Yun Liang, Shuo Wang, Renze Chen, Kaiwen Sheng. _Peking University_.
    - [Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation](https://cseweb.ucsd.edu/~bhahn221/doc/paper/iclr20-chameleon.pdf) **ICLR 2020**. [code](https://github.com/anony-sub/chameleon). Byung Hoon Ahn, Prannoy Pilligundla, Amir Yazdanbakhsh, Hadi Esmaeilzadeh. _University of California, San Diego_.

- Dynamic Shape and Control Flow Optimization


## Compilers for Configurable Hardware


## Survey and Books

## Talks, Tutorials, and Videos