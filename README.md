## Compilers and Architectures
![GitHub last commit](https://img.shields.io/github/last-commit/KnowingNothing/compiler-and-arch)
[![](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A list of awesome compilers and optimization techniques (applicable to compilers) for different architectures and emerging domains.

>Note: Although some projects are not about compiler design or implementation themselves, we still include them once their techniques are suitable for automation and compiler design.

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
- [Design Space Construction and Exploration](#design-space-construction-and-exploration)
- [Dynamic Shape and Control Flow](#dynamic-shape-and-control-flow)
- [Sparse Applications, Compilers, and Architectures
](#sparse-applications-compilers-and-architectures)
- [Tree and Graph Applications, Compilers, and Architectures](#tree-and-graph-applications-compilers-and-architectures)
- [NAS Compilers and Architectures](#nas-compilers-and-architectures)
- [Survey and Books](#survey-and-books)
- [Talks, Tutorials, and Videos](#talks-tutorials-and-videos)


## List of Conferences and Journals Considered
- Conferences
    - ASPLOS, ISCA, MICRO, HPCA
    - OSDI, SOSP, PLDI, PPoPP, SC
    - DAC, ICLR, NeurIPS, ATC, OOPSLA
    - CGO, MLSys, SIGGRAPH, PACT, POPL, ICS
    - Euro-Par
- Journals
    - TPDS, TCAD, TC
    - TACO
- Preprint
    - arXiv


## Compiler Toolchain

- Open-source
    - [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054) **arXiv 2020**, [Github Page](https://github.com/llvm/llvm-project.git). [Document Page](https://mlir.llvm.org/). Chris Lattner, Jacques A. Pienaar, Mehdi Amini, Uday Bondhugula, River Riddle, Albert Cohen, Tatiana Shpeisman, Andy Davis, Nicolas Vasilache, Oleksandr Zinenko. _Google_.
    - [JAX: Compiling machine learning programs via high-level tracing](https://cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf) **MLSys 2018**. [Github Page](https://github.com/google/jax). [Document Page](https://jax.readthedocs.io/en/latest/). Roy Frostig, Matthew James Johnson, and Chris Leary. _Google_.
    - [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) **OSDI 2018**. [Github Page](https://github.com/apache/tvm). [Document Page](https://tvm.apache.org/). Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Q. Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy. _University of Washington_.
    - [XLA: Optimizing Compiler for Machine Learning](https://www.tensorflow.org/xla). _Google_.
    - [Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) **PLDI 2013**. [Github Page](https://github.com/halide/Halide). [Document Page](https://halide-lang.org/docs/). Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, Saman P. Amarasinghe. _MIT CSAIL_.
    - [The Tensor Algebra Compiler](https://dl.acm.org/doi/10.1145/3133901) **OOPSLA 2017**. [Github Page](https://github.com/tensor-compiler/taco). [Document Page](http://tensor-compiler.org/). Fredrik Kjolstad, Shoaib Kamil, Stephen Chou, David Lugato, Saman P. Amarasinghe. _Massachusetts Institute of Technology_.

- Close-source (binary available)


## Compilers for AI chips

- Auto-tensorization and Auto-vectorization
    - [TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://arxiv.org/abs/2207.04296). **arXiv 2022**. Siyuan Feng, Bohan Hou, Hongyi Jin, Wuwei Lin, Junru Shao, Ruihang Lai, Zihao Ye, Lianmin Zheng, Cody Hao Yu, Yong Yu, Tianqi Chen. _Shanghai Jiao Tong University_.
    - [AMOS: enabling automatic mapping for tensor computations on spatial accelerators with hardware abstraction](https://dl.acm.org/doi/abs/10.1145/3470496.3527440) **ISCA 2022**. [code](https://github.com/KnowingNothing/AMOS). Size Zheng, Renze Chen, Anjiang Wei, Yicheng Jin, Qin Han, Liqiang Lu, Bingyang Wu, Xiuhong Li, Shengen Yan, Yun Liang. _Peking University_.
    - [UNIT: Unifying Tensorized Instruction Compilation](https://polyarch.cs.ucla.edu/papers/cgo2021-unit.pdf) **CGO 2021**. [code](https://github.com/were/UNIT). Jian Weng, Animesh Jain, Jie Wang, Leyuan Wang, Yida Wang, Tony Nowatzki. _University of California, Los Angeles_.
    - [VeGen: A Vectorizer Generator for SIMD and Beyond](https://dl.acm.org/doi/10.1145/3445814.3446692) **ASPLOS 2021**. [code](https://github.com/ychen306/vegen). Yishen Chen, Charith Mendis, Michael Carbin, Saman P. Amarasinghe. _MIT CSAIL_.

- Polyhedral Optimization
    - [AKG: automatic kernel generation for neural processing units using polyhedral transformations](https://dl.acm.org/doi/abs/10.1145/3453483.3454106) **PLDI 2021**. [code](https://github.com/mindspore-ai/akg). Jie Zhao, Bojie Li, Wang Nie, Zhen Geng, Renwei Zhang, Xiong Gao, Bin Cheng, Chen Wu, Yun Cheng, Zheng Li, Peng Di, Kun Zhang, Xuefeng Jin. _State Key Laboratory of Mathematical Engineering and Advanced Computing, China_.
    - [Optimizing the Memory Hierarchy by Compositing Automatic Transformations on Computations and Data](https://www.microarch.org/micro53/papers/738300a427.pdf) **MICRO 2020**. [code](https://github.com/mindspore-ai/akg). Jie Zhao, Peng Di. _State Key Laboratory of Mathematical Engineering and Advanced Computing, China_.
    - [Hardware Abstractions for targeting EDDO Architectures with the Polyhedral Model](http://poantsai.me/papers/2021.polyEDDO.impact.pdf) **PACT 2021**. Angshuman Parashar, Prasanth Chatarasi, Po-An Tsai. _NVIDIA_.

## Compilers for PIM


## Compilers for Brain-inspired Hardware


## Compilers for SIMT GPU
- Efficient Compute-intensive Kernel Generation
    - [Roller: Fast and Efficient Tensor Compilation for Deep Learning](https://www.usenix.org/conference/osdi22/presentation/zhu) **OSDI 2022**. Hongyu Zhu, Ruofan Wu, Yijia Diao, Shanbin Ke, Haoyu Li, Chen Zhang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Wei Cui, Fan Yang, Mao Yang, and Lidong Zhou, Asaf Cidon, Gennady Pekhimenko. _University of Toronto and Microsoft Research_.
    - [Automatic Kernel Generation for Volta Tensor Cores](https://arxiv.org/abs/2006.12645) **arXiv 2020**. Somashekaracharya G. Bhaskaracharya, Julien Demouth, Vinod Grover. _NVIDIA_.

- Efficient Compute-intensive Kernel Fusion
    - [Tacker: Tensor-CUDA Core Kernel Fusion for Improving the GPU Utilization while Ensuring QoS](https://ieeexplore.ieee.org/document/9773253) **HPCA 2022**. [code](https://github.com/sjtu-epcc/Tacker). Han Zhao, Weihao Cui, Quan Chen, Youtao Zhang, Yanchao Lu, Chao Li, Jingwen Leng, Minyi Guo. _Shanghai Jiao Tong University_.
    - [BOLT: BRIDGING THE GAP BETWEEN AUTO-TUNERS AND HARDWARE-NATIVE PERFORMANCE](https://jxing.me/pdf/bolt-mlsys21.pdf) **MLSys 2022**. [code](https://github.com/apache/tvm/commit/541f9f2d8aef9697fd7ccb6a7c0644da273f33b6). Jiarong Xing, Leyuan Wang, Shang Zhang, Jack Chen, Ang Chen, Yibo Zhu. _Rice University_.
    - [Accelerating Deep Learning Inference with Cross-Layer Data Reuse on GPUs](https://arxiv.org/abs/2007.06000) **Euro-Par 2020**. Xueying Wang, Guangli Li, Xiao Dong, Jiansong Li, Lei Liu, Xiaobing Feng. _Institute of Computing Technology, Chinese Academy of Science_.

- Efficient Memory-intensive Kernel Fusion
    - [Automatic Horizontal Fusion for GPU Kernels](https://aoli.al/papers/hfuse-cgo22.pdf) **CGO 2022**. Ao Li, Bojian Zheng, Gennady Pekhimenko, Fan Long. _Carnegie Mellon University_.
    - [AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-Intensive ML Training and Inference on Modern SIMT Architectures](https://dl.acm.org/doi/10.1145/3503222.3507723) **ASPLOS 2022**. Zhen Zheng, Xuanda Yang, Pengzhan Zhao, Guoping Long, Kai Zhu, Feiwen Zhu, Wenyi Zhao, Xiaoyong Liu, Jun Yang, Jidong Zhai, Shuaiwen Leon Song, Wei Lin. _Alibaba Group_.
    - [FusionStitching: Boosting Memory Intensive Computations for Deep Learning Workloads](https://arxiv.org/abs/2009.10924) **arXiv 2020**. Zhen Zheng, Pengzhan Zhao, Guoping Long, Feiwen Zhu, Kai Zhu, Wenyi Zhao, Lansong Diao, Jun Yang, Wei Lin. _Alibaba Group_.
    - [From Loop Fusion to Kernel Fusion: A Domain-Specific Approach to Locality Optimization](https://ieeexplore.ieee.org/document/8661176) **CGO 2019**. Bo Qiao, Oliver Reiche, Frank Hannig, Jürgen Teich. _Friedrich-Alexander University Erlangen-Nürnberg_.
    - [Versapipe: a versatile programming framework for pipelined computing on GPU](https://people.engr.ncsu.edu/xshen5/Publications/micro17a.pdf) **MICRO 2017**. [cdoe](https://github.com/JamesTheZ/VersaPipe).	Zhen Zheng, Chanyoung Oh, Jidong Zhai, Xipeng Shen, Youngmin Yi, Wenguang Chen. _Tsinghua University_.
    - [Scalable Kernel Fusion for Memory-Bound GPU Applications](https://ieeexplore.ieee.org/document/7013003) **SC 2014**. Mohamed Wahib, Naoya Maruyama. _RIKEN Advanced Institute for Computational Science JST, CREST_.

- Polyhedral Optimization
    - [Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://arxiv.org/abs/1802.04730) **arXiv 2018**. [code](https://github.com/facebookresearch/TensorComprehensions). Nicolas Vasilache, Oleksandr Zinenko, Theodoros Theodoridis, Priya Goyal, Zachary DeVito, William S. Moses, Sven Verdoolaege, Andrew Adams, Albert Cohen. _Facebook AI Research_.

- Program Synthesis
    - [EQUALITY SATURATION FOR TENSOR GRAPH SUPEROPTIMIZATION](https://arxiv.org/abs/2101.01332) **MLSys 2021**. Yichen Yang, Phitchaya Mangpo Phothilimthana, Yisu Remy Wang, Max Willsey, Sudip Roy, Jacques Pienaar. _MIT EECS & CSAIL_.
    - [Swizzle Inventor: Data Movement Synthesis for GPU Kernels](https://homes.cs.washington.edu/~emina/doc/swizzle.asplos19.pdf) **ASPLOS 2019**. Phitchaya Mangpo Phothilimthana, Archibald Samuel Elliott, An Wang, Abhinav Jangda, Bastian Hagedorn, Henrik Barthels, Samuel J. Kaufman, Vinod Grover, Emina Torlak, Rastislav Bodík. _University of California, Berkeley_.

- Compilers for Irregular Workloads
    - [FreeTensor: A Free-Form DSL with Holistic Optimizations for Irregular Tensor Programs](https://pacman.cs.tsinghua.edu.cn/~zjd/publication/pldi22-freetensor/pldi22-freetensor.pdf) **PLDI 2022**. [code](https://github.com/roastduck/FreeTensor). Shizhi Tang, Jidong Zhai, Haojie Wang, Lin Jiang, Liyan Zheng, Zhenhao Yuan, Chen Zhang. _Tsinghua University_.

- Compilers for HPC Workloads on GPU
    - [Domain-Specific Multi-Level IR Rewriting for GPU: The Open Earth Compiler for GPU-accelerated Climate Simulation](https://dl.acm.org/doi/10.1145/3469030) **TACO 2021**. [code](https://github.com/spcl/open-earth-compiler). Tobias Gysi, Christoph Müller, Oleksandr Zinenko, Stephan Herhut, Eddie Davis, Tobias Wicky, Oliver Fuhrer, Torsten Hoefler, Tobias Grosser. _ETH Zurich_.

- Holistic Graph Optimization
    - [APOLLO: AUTOMATIC PARTITION-BASED OPERATOR FUSION THROUGH LAYER BY LAYER OPTIMIZATION](https://proceedings.mlsys.org/paper/2022/hash/069059b7ef840f0c74a814ec9237b6ec-Abstract.html) **MLSys 2022**. Jie Zhao, Xiong Gao, Ruijie Xia, Zhaochuang Zhang, Deshi Chen, Lei Chen, Renwei Zhang, Zhen Geng, Bin Cheng, Xuefeng Jin. _State Key Laboratory of Mathematical Engineering and Advanced Computing_.
    - [NeoFlow: A Flexible Framework for Enabling Efficient Compilation for High Performance DNN Training](https://www.computer.org/csdl/journal/td/2022/11/09664259/1zHDLHIxWPC) **TPDS 2022**. Size Zheng, Renze Chen, Yicheng Jin, Anjiang Wei, Bingyang Wu, Xiuhong Li, Shengen Yan, Yun Liang. _Peking University_.
    - [DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion](https://dl.acm.org/doi/10.1145/3453483.3454083) **PLDI 2021**. Wei Niu, Jiexiong Guan, Yanzhi Wang, Gagan Agrawal, Bin Ren. _College of William & Mary_.
    - [DeepCuts: A Deep Learning Optimization Framework for Versatile GPU Workloads](https://dl.acm.org/doi/10.1145/3453483.3454038) **PLDI 2021** Wookeun Jung, Thanh Tuan Dao, Jaejin Lee. _Seoul National University_.
    - [Pet: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections](https://www.usenix.org/conference/osdi21/presentation/wang#:~:text=We%20propose%20PET%2C%20the%20first,only%20maintain%20partial%20functional%20equivalence.) **OSDI 2021**. [code](https://github.com/thu-pacman/PET). Haojie Wang, Jidong Zhai, Mingyu Gao, Zixuan Ma, Shizhi Tang, Liyan Zheng, Yuanzhi Li, Kaiyuan Rong, Yuanyong Chen, Zhihao Jia. _Tsinghua University_.
    - [Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks](https://www.usenix.org/conference/osdi20/presentation/ma) **OSDI 2020**. [code](https://github.com/microsoft/nnfusion). Lingxiao Ma, Zhiqiang Xie, Zhi Yang, Jilong Xue, Youshan Miao, Wei Cui, Wenxiang Hu, Fan Yang, Lintao Zhang, Lidong Zhou. _Peking University and Microsoft Research_.
    - [TASO: optimizing deep learning computation with automatic generation of graph substitutions](https://cs.stanford.edu/~padon/taso-sosp19.pdf) **SOSP 2019**. [code](https://github.com/jiazhihao/TASO). Zhihao Jia, Oded Padon, James J. Thomas, Todd Warszawski, Matei Zaharia, Alex Aiken. _Stanford University_.

- Distributed Optimization
    - [DISTAL: The Distributed Tensor Algebra Compiler](http://tensor-compiler.org/files/yadav-pldi22-distal.pdf) **PLDI 2022**. Rohan Yadav, Alex Aiken, and Fredrik Kjolstad. _Stanford University_.
    - [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023) **OSDI 2022**. Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Joseph E. Gonzalez, Ion Stoica. _UC Berkeley_
    - [VirtualFlow: Decoupling Deep Learning Models from the Underlying Hardware](https://proceedings.mlsys.org/paper/2022/hash/2723d092b63885e0d7c260cc007e8b9d-Abstract.html) **MLSys 2022**. Andrew Or, Haoyu Zhang, Michael None Freedman. _Princeton University_.
    - [OneFlow: Redesign the Distributed Deep Learning Framework from Scratch](https://arxiv.org/abs/2110.15032) **arXiv 2021**. Jinhui Yuan, Xinqi Li, Cheng Cheng, Juncheng Liu, Ran Guo, Shenghang Cai, Chi Yao, Fei Yang, Xiaodong Yi, Chuan Wu, Haoran Zhang, Jie Zhao. _OneFlow Research_.


## Compilers for CPU


## Compilers for Mobile and Edge


## Compilers for RISC-V


## Compilers for Configurable Hardware


## Design Space Construction and Exploration

- Auto-tuning and Auto-scheduling
    - [Efficient Automatic Scheduling of Imaging and Vision Pipelines for the GPU](https://cseweb.ucsd.edu/~tzli/gpu_autoscheduler.pdf) **OOPSLA 2021**. Luke Anderson, Andrew Adams, Karima Ma, Tzu-Mao Li, Tian Jin, Jonathan Ragan-Kelley. _Massachusetts Institute of Technology_.
    - [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/system/files/osdi20-zheng.pdf) **OSDI 2020**. [code](https://github.com/apache/tvm/tree/main/python/tvm/auto_scheduler). Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, Ion Stoica. _UC Berkeley_.
    - [FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System](https://dl.acm.org/doi/10.1145/3373376.3378508) **ASPLOS 2020**. [code](https://github.com/KnowingNothing/FlexTensor). Size Zheng, Yun Liang, Shuo Wang, Renze Chen, Kaiwen Sheng. _Peking University_.
    - [ProTuner: Tuning Programs with Monte Carlo Tree Search](https://arxiv.org/abs/2005.13685) **arXiv 2020**. Ameer Haj-Ali, Hasan Genc, Qijing Huang, William S. Moses, John Wawrzynek, Krste Asanovic, Ion Stoica. _UC Berkeley_.
    - [Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation](https://cseweb.ucsd.edu/~bhahn221/doc/paper/iclr20-chameleon.pdf) **ICLR 2020**. [code](https://github.com/anony-sub/chameleon). Byung Hoon Ahn, Prannoy Pilligundla, Amir Yazdanbakhsh, Hadi Esmaeilzadeh. _University of California, San Diego_.
    - [Learning to Optimize Halide with Tree Search and Random Programs](https://dl.acm.org/doi/10.1145/3306346.3322967) **SIGGRAPH 2019**. Andrew Adams, Karima Ma, Luke Anderson, Riyadh Baghdadi, Tzu-Mao Li, Michaël Gharbi, Benoit Steiner, Steven Johnson, Kayvon Fatahalian, Frédo Durand, Jonathan Ragan-Kelley. _Facebook AI Research_.
    - [Learning to Optimize Tensor Programs](https://proceedings.neurips.cc/paper/2018/file/8b5700012be65c9da25f49408d959ca0-Paper.pdf) **NeurIPS 2018**. [code](https://github.com/apache/tvm/tree/main/python/tvm/autotvm). Tianqi Chen, Lianmin Zheng, Eddie Q. Yan, Ziheng Jiang, Thierry Moreau, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy. _University of Washington_.
    - [Automatically Scheduling Halide Image Processing Pipelines](https://dl.acm.org/doi/pdf/10.1145/2897824.2925952) **SIGGRAPH 2016**. Ravi Teja Mullapudi, Andrew Adams, Dillon Sharlet, Jonathan Ragan-Kelley, Kayvon Fatahalian. _Carnegie Mellon University_.

- Analytical Approaches
    - [Analytical characterization and design space exploration for optimization of CNNs](https://dl.acm.org/doi/10.1145/3445814.3446759) **ASPLOS 2021**. Rui Li, Yufan Xu, Aravind Sukumaran-Rajam, Atanas Rountev, P. Sadayappan. _University of Utah_.
    - [Tuna: A Static Analysis Approach to Optimizing Deep Neural Networks](https://arxiv.org/abs/2104.14641#:~:text=Tuna%3A%20A%20Static%20Analysis%20Approach%20to%20Optimizing%20Deep%20Neural%20Networks,-Yao%20Wang%2C%20Xingyu&text=We%20introduce%20Tuna%2C%20a%20static,performance%20of%20deep%20neural%20networks.) **arXiv 2021**. Yao Wang, Xingyu Zhou, Yanming Wang, Rui Li, Yong Wu, Vin Sharma. _Amazon Web Services_.
    - [Analytical cache modeling and tilesize optimization for tensor contractions](https://hal.inria.fr/hal-02418875/document) **SC 2019**. 	Rui Li, Aravind Sukumaran-Rajam, Richard Veras, Tze Meng Low, Fabrice Rastello, Atanas Rountev, P. Sadayappan. _University of Utah_.


## Dynamic Shape and Control Flow


## Sparse Applications, Compilers, and Architectures
- Compiler Design
    - [Autoscheduling for Sparse Tensor Algebra with an Asymptotic Cost Model](http://tensor-compiler.org/files/ahrens-pldi22-taco-autoscheduling.pdf) **PLDI 2022**. Peter Ahrens, Fredrik Kjolstad, and Saman Amarasinghe. _MIT CSAIL_.
    - [Unified Compilation for Lossless Compression and Sparse Computing](https://groups.csail.mit.edu/commit/papers/2022/cgo22-TACO-lossless.pdf) **CGO 2022**. Daniel Donenfeld, Stephen Chou, and Saman Amarasinghe. _MIT CSAIL_.
    - [Dynamic Sparse Tensor Algebra Compilation](https://arxiv.org/abs/2112.01394) **arXiv 2021**. Stephen Chou and Saman Amarasinghe. _MIT CSAIL_.
    - [Compilation of Sparse Array Programming Models](https://dl.acm.org/doi/abs/10.1145/3485505) **OOPSLA 2021**. Rawn Henry, Olivia Hsu, Rohan Yadav, Stephen Chou, Kunle Olukotun, Saman Amarasinghe, and Fredrik Kjolstad. _MIT CSAIL_.
    - [A sparse iteration space transformation framework for sparse tensor algebra](https://dl.acm.org/doi/10.1145/3428226) **OOPSLA 2020**. Ryan Senanayake, Changwan Hong, Ziheng Wang, Amalee Wilson, Stephen Chou, Shoaib Kamil, Saman P. Amarasinghe, Fredrik Kjolstad. _Reservoir Labs_.
    - [Automatic Generation of Efficient Sparse Tensor Format Conversion Routines](https://arxiv.org/pdf/2001.02609.pdf) **PLDI 2020**. Stephen Chou, Fredrik Kjolstad, and Saman Amarasinghe. _MIT CSAIL_.
    - [Tensor Algebra Compilation with Workspaces](http://fredrikbk.com/publications/taco-workspaces.pdf) **CGO 2019**. Fredrik Kjolstad, Peter Ahrens, Shoaib Kamil, Saman P. Amarasinghe. _MIT_.

## Tree and Graph Applications, Compilers, and Architectures
- Compiler Design
    - [GraphIt to CUDA Compiler in 2021 LOC: A Case for High-Performance DSL Implementation via Staging with BuilDSL](https://intimeand.space/docs/CGO2022-BuilDSL.pdf) **CGO 2022**. Ajay Brahmakshatriya, Saman P. Amarasinghe. _CSAIL, MIT_.
    - [Taming the Zoo: The Unified GraphIt Compiler Framework for Novel Architectures](https://ieeexplore.ieee.org/document/9499863) **ISCA 2021**. Ajay Brahmakshatriya, Emily Furst, Victor A. Ying, Claire Hsu, Changwan Hong, Max Ruttenberg, Yunming Zhang, Dai Cheol Jung, Dustin Richmond, Michael B. Taylor, Julian Shun, Mark Oskin, Daniel Sánchez, Saman P. Amarasinghe. _MIT CSAIL_.
    - [A Tensor Compiler for Unified Machine Learning Prediction Serving](https://www.usenix.org/conference/osdi20/presentation/nakandala) **OSDI 2020**. [code](https://github.com/microsoft/hummingbird). Supun Nakandala, Karla Saur, Gyeong-In Yu, Konstantinos Karanasos, Carlo Curino, Markus Weimer, Matteo Interlandi. _UC San Diego_.
    - [Optimizing ordered graph algorithms with GraphIt](https://arxiv.org/abs/1911.07260) **CGO 2020**. Yunming Zhang, Ajay Brahmakshatriya, Xinyi Chen, Laxman Dhulipala, Shoaib Kamil, Saman P. Amarasinghe, Julian Shun. _MIT CSAIL_.
    - [GraphIt: A High-Performance Graph DSL](https://dl.acm.org/doi/10.1145/3276491) **OOPSLA 2018**. Yunming Zhang, Mengjiao Yang, Riyadh Baghdadi, Shoaib Kamil, Julian Shun, Saman P. Amarasinghe. _MIT CSAIL_.


## NAS Compilers and Architectures
- Compiler Design
    - [Neural Architecture Search as Program Transformation Exploration](https://dl.acm.org/doi/abs/10.1145/3445814.3446753) **ASPLOS 2021**. Jack Turner, Elliot J. Crowley, Michael F. P. O'Boyle. _University of Edinburgh United Kingdom_.


## Cost Model
- Model Design
    - [A LEARNED PERFORMANCE MODEL FOR TENSOR PROCESSING UNITS](https://arxiv.org/abs/2008.01040) **MLSys 2021**. Samuel J. Kaufman, Phitchaya Mangpo Phothilimthana, Yanqi Zhou, Charith Mendis, Sudip Roy, Amit Sabne, Mike Burrows. _Google_.

- Dataset
    - [TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf) **NeurIPS Datasets and Benchmarks 2021**. [code](https://github.com/tlc-pack/tenset). Lianmin Zheng, Ruochen Liu, Junru Shao, Tianqi Chen, Joseph Gonzalez, Ion Stoica, Ameer Haj-Ali. _UC Berkeley_.


## Survey and Books

## Talks, Tutorials, and Videos