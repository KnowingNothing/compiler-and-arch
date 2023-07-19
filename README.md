## Compilers and Architectures
![GitHub last commit](https://img.shields.io/github/last-commit/KnowingNothing/compiler-and-arch)
[![](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A list of awesome compilers and optimization techniques (applicable to compilers) for different architectures and emerging domains.

>Note: Although some projects are not about compiler design or implementation themselves, we still include them if their techniques are suitable for automation and compiler design.

## Contents


- [Compilers and Architectures](#compilers-and-architectures)
- [Contents](#contents)
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
- [Sparse Applications, Compilers, and Architectures](#sparse-applications-compilers-and-architectures)
- [Tree and Graph Applications, Compilers, and Architectures](#tree-and-graph-applications-compilers-and-architectures)
- [NAS Compilers and Architectures](#nas-compilers-and-architectures)
- [Security and Privacy](#security-and-privacy)
- [Cost Model](#cost-model)
- [Hardware Design and Validation](#hardware-design-and-validation)
- [Survey and Books](#survey-and-books)
- [Talks, Tutorials, and Videos](#talks-tutorials-and-videos)



## List of Conferences and Journals Considered
- Conferences
    - ASPLOS, ISCA, MICRO, HPCA
    - OSDI, SOSP, PLDI, PPoPP, SC
    - DAC, ICLR, NeurIPS, ATC, OOPSLA
    - CGO, MLSys, SIGGRAPH, PACT, POPL, ICS
    - Euro-Par, MAPL
    - ICRC
- Journals
    - TPDS, TCAD, TC
    - TACO, TECS
- Preprint
    - arXiv


## Compiler Toolchain

- Open-source
    - [A Data-Centric Optimization Framework for Machine Learning](https://arxiv.org/pdf/2110.10802.pdf#:~:text=The%20Data%2DCentric%20Machine%20Learning,art%20performance%20for%20multiple%20platforms.) **ICS 2022**. [Github Page](https://github.com/spcl/daceml). [Document Page](https://daceml.readthedocs.io/en/latest/?badge=latest). Oliver Rausch, Tal Ben-Nun, Nikoli Dryden, Andrei Ivanov, Shigang Li, Torsten Hoefler. _ETH Zurich_.
    - [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054) **arXiv 2020**, [Github Page](https://github.com/llvm/llvm-project.git). [Document Page](https://mlir.llvm.org/). Chris Lattner, Jacques A. Pienaar, Mehdi Amini, Uday Bondhugula, River Riddle, Albert Cohen, Tatiana Shpeisman, Andy Davis, Nicolas Vasilache, Oleksandr Zinenko. _Google_.
    - [JAX: Compiling machine learning programs via high-level tracing](https://cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf) **MLSys 2018**. [Github Page](https://github.com/google/jax). [Document Page](https://jax.readthedocs.io/en/latest/). Roy Frostig, Matthew James Johnson, and Chris Leary. _Google_.
    - [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) **OSDI 2018**. [Github Page](https://github.com/apache/tvm). [Document Page](https://tvm.apache.org/). Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Q. Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy. _University of Washington_.
    - [XLA: Optimizing Compiler for Machine Learning](https://www.tensorflow.org/xla). _Google_.
    - [Intel nGraph: An Intermediate Representation, Compiler, and Executor for Deep Learning](https://arxiv.org/abs/1801.08058) **arXiv 2018**. [Github Page](https://github.com/NervanaSystems/ngraph). [Document Page](https://github.com/openvinotoolkit/openvino_notebooks). Scott Cyphers, Arjun K. Bansal, Anahita Bhiwandiwalla, Jayaram Bobba, Matthew Brookhart, Avijit Chakraborty, William Constable, Christian Convey, Leona Cook, Omar Kanawi, Robert Kimball, Jason Knight, Nikolay Korovaiko, Varun Kumar Vijay, Yixing Lao, Christopher R. Lishka, Jaikrishnan Menon, Jennifer Myers, Sandeep Aswath Narayana, Adam Procter, Tristan J. Webb. _Intel_.
    - [Glow: Graph Lowering Compiler Techniques for Neural Networks](https://arxiv.org/pdf/1805.00907.pdf) **arXiv 2018**. [Github Page](https://github.com/pytorch/glow). Nadav Rotem, Jordan Fix, Saleem Abdulrasool, Summer Deng, Roman Dzhabarov, James Hegeman, Roman Levenstein, Bert Maher, Nadathur Satish, Jakob Olesen, Jongsoo Park, Artem Rakhov, Misha Smelyanskiy. _Facebook_.
    - [DLVM: A modern compiler infrastructure for deep learning systems](https://arxiv.org/abs/1711.03016) **ICLR 2018**. [Github Page](https://github.com/dlvm-team). Richard Wei, Lane Schwartz, Vikram S. Adve. _University of Illinois at Urbana-Champaign_.
    - [The Tensor Algebra Compiler](https://dl.acm.org/doi/10.1145/3133901) **OOPSLA 2017**. [Github Page](https://github.com/tensor-compiler/taco). [Document Page](http://tensor-compiler.org/). Fredrik Kjolstad, Shoaib Kamil, Stephen Chou, David Lugato, Saman P. Amarasinghe. _Massachusetts Institute of Technology_.
    - [Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) **PLDI 2013**. [Github Page](https://github.com/halide/Halide). [Document Page](https://halide-lang.org/docs/). Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, Saman P. Amarasinghe. _MIT CSAIL_.

- Close-source (binary available)
    - [TensorRT](https://developer.nvidia.com/tensorrt) with some components open-source at [Github](https://github.com/NVIDIA/TensorRT).


## Compilers for AI chips

- Auto-tensorization and Auto-vectorization
    - [Graphene: An IR for Optimized Tensor Computations on GPUs](https://dl.acm.org/doi/abs/10.1145/3582016.3582018) **ASPLOS 2023**. Bastian Hagedorn, Bin Fan, Hanfeng Chen, Cris Cecka, Michael Garland, Vinod Grover. _NVIDIA_.
    - [TensorIR: An Abstraction for Automatic Tensorized Program Optimization](https://arxiv.org/abs/2207.04296). **ASPLOS 2023**. Siyuan Feng, Bohan Hou, Hongyi Jin, Wuwei Lin, Junru Shao, Ruihang Lai, Zihao Ye, Lianmin Zheng, Cody Hao Yu, Yong Yu, Tianqi Chen. _Shanghai Jiao Tong University_.
    - [Heron: Automatically Constrained High-Performance Library Generation for Deep Learning Accelerators](https://dl.acm.org/doi/abs/10.1145/3582016.3582061). **ASPLOS 2023**. Jun Bi, Qi Guo, Xiaqing Li, Yongwei Zhao, Yuanbo Wen, Yuxuan Guo, Enshuai Zhou, Xing Hu, Zidong Du, Ling Li, Huaping Chen, Tianshi Chen. _University of Science and Technology of China_.
    - [Exocompilation for productive programming of hardware accelerators](https://dl.acm.org/doi/pdf/10.1145/3519939.3523446). **PLDI 2022**. [code](https://github.com/exo-lang/exo). Yuka Ikarashi, Gilbert Louis Bernstein, Alex Reinking, Hasan Genc, Jonathan Ragan-Kelley. _MIT_.
    - [AMOS: enabling automatic mapping for tensor computations on spatial accelerators with hardware abstraction](https://dl.acm.org/doi/abs/10.1145/3470496.3527440) **ISCA 2022**. [code](https://github.com/KnowingNothing/AMOS). Size Zheng, Renze Chen, Anjiang Wei, Yicheng Jin, Qin Han, Liqiang Lu, Bingyang Wu, Xiuhong Li, Shengen Yan, Yun Liang. _Peking University_.
    - [UNIT: Unifying Tensorized Instruction Compilation](https://polyarch.cs.ucla.edu/papers/cgo2021-unit.pdf) **CGO 2021**. [code](https://github.com/were/UNIT). Jian Weng, Animesh Jain, Jie Wang, Leyuan Wang, Yida Wang, Tony Nowatzki. _University of California, Los Angeles_.

- Polyhedral Optimization
    - [AKG: automatic kernel generation for neural processing units using polyhedral transformations](https://dl.acm.org/doi/abs/10.1145/3453483.3454106) **PLDI 2021**. [code](https://github.com/mindspore-ai/akg). Jie Zhao, Bojie Li, Wang Nie, Zhen Geng, Renwei Zhang, Xiong Gao, Bin Cheng, Chen Wu, Yun Cheng, Zheng Li, Peng Di, Kun Zhang, Xuefeng Jin. _State Key Laboratory of Mathematical Engineering and Advanced Computing, China_.
    - [Optimizing the Memory Hierarchy by Compositing Automatic Transformations on Computations and Data](https://www.microarch.org/micro53/papers/738300a427.pdf) **MICRO 2020**. [code](https://github.com/mindspore-ai/akg). Jie Zhao, Peng Di. _State Key Laboratory of Mathematical Engineering and Advanced Computing, China_.
    - [Hardware Abstractions for targeting EDDO Architectures with the Polyhedral Model](http://poantsai.me/papers/2021.polyEDDO.impact.pdf) **PACT 2021**. Angshuman Parashar, Prasanth Chatarasi, Po-An Tsai. _NVIDIA_.

## Compilers for PIM
- Code Generation
    - [CHOPPER: A Compiler Infrastructure for Programmable Bit-serial SIMD Processing Using Memory in DRAM](https://ieeexplore.ieee.org/document/10071070) **HPCA 2023**. Xiangjun Peng, Yaohua Wang,Ming-Chang Yang. _The Chinese University of Hong Kong_.
- Software & Hardware Co-Design
    - [Hardware-Software Co-Design for an Analog-Digital Accelerator for Machine Learning](http://sitaohuang.com/publications/2018_icrc_analog_ml.pdf) **ICRC 2018**. Ambrosi, Joao and Ankit, Aayush and Antunes, Rodrigo and Chalamalasetti, Sai Rahul and Chatterjee, Soumitra and El Hajj, Izzat and Fachini, Guilherme and Faraboschi, Paolo and Foltin, Martin and Huang, Sitao. _Hewlett Packard Enterprise_.
    - [SIAM: Chiplet-based Scalable In-Memory Acceleration with Mesh for Deep Neural Networks](https://arxiv.org/pdf/2108.08903.pdf) **TECS 2021**. [code](https://github.com/gkrish19/SIAM-Chiplet-based-Scalable-In-Memory-Acceleration-with-Mesh-for-Deep-Neural-Networks). Krishnan, Gokul and Mandal, Sumit K and Pannala, Manvitha and Chakrabarti, Chaitali and Seo, Jae-Sun and Ogras, Umit Y and Cao, Yu. _Arizona State University_.
    - [FPSA: A Full System Stack Solution for Reconfigurable ReRAM-based NN Accelerator Architecture](https://dl.acm.org/doi/pdf/10.1145/3297858.3304048) **ASPLOS 2019**. Yu Ji, Youyang Zhang, Xinfeng Xie, Shuangchen Li, Peiqi Wang, Xing Hu, Youhui Zhang, Yuan Xie. _Tsinghua University_

- End2End Compiler
    -[PUMA: A Programmable Ultra-efficient Memristor-based Accelerator for Machine Learning Inference](https://dl.acm.org/doi/10.1145/3297858.3304049) **ASPLOS 2019**. [code](https://github.com/illinois-impact/puma-compiler) A. Ankit, I. El Hajj, S. Chalamalasetti, G. Ndu, M. Foltin, R. S. Williams, P. Faraboschi, W.-M. Hwu, J. P. Strachan, K. Roy, D. Milojicic. _Purdue University_
    - [OCC: An Automated End-to-End Machine Learning Optimizing Compiler for Computing-In-Memory](https://grosser.science/static/bd7e99f44a1eb746980f3c84ec961858/2107_Khan_TCAD.pdf) **TCAD 2021**. [code](https://github.com/adam-smnk/Open-CIM-Compiler). Siemieniuk, Adam and Chelini, Lorenzo and Khan, Asif Ali and Castrillon, Jeronimo and Drebes, Andi and Corporaal, Henk and Grosser, Tobias and Kong, Martin.
    -[Polyhedral-Based Compilation Framework for In-Memory Neural Network Accelerators](https://dl.acm.org/doi/10.1145/3469847) **JETCS 2021**. [code](https://github.com/Jianhui-Han/polyxb/tree/22acd3059a3712d63802ecf100a3ec2973c9b8ab). Jianhui Han, Xiang Fei, Zhaolin Li, Youhui Zhang. _Tsinghua University_

- Code Offloading, Mapping and Scheduling
    - [Scheduling Techniques for GPU Architectures with Processing-In-Memory Capabilities](https://dl.acm.org/doi/pdf/10.1145/2967938.2967940) **PACT 2016**. Ashutosh Pattnaik, Xulong Tang, Adwait Jog, Onur Kayiran, Asit K. Mishra, Mahmut T. Kandemir, Onur Mutlu, Chita R. Das. _Pennsylvania State University_
    - [Transparent Offloading and Mapping (TOM): Enabling Programmer-Transparent Near-Data Processing in GPU Systems](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7551394) **ISCA 2016**. 	Kevin Hsieh, Eiman Ebrahimi, Gwangsun Kim, Niladrish Chatterjee, Mike O'Connor, Nandita Vijaykumar, Onur Mutlu, Stephen W. Keckler. _Carnegie Mellon University_
    - [To PIM or not for emerging general purpose processing in DDR memory systems](https://dl.acm.org/doi/pdf/10.1145/3470496.3527431) **ISCA 2022**. Alexandar Devic, Siddhartha Balakrishna Rai, Anand Sivasubramaniam, Ameen Akel, Sean Eilert, Justin Eno. _The Pennsylvania State University_ 

- Synthesis
    - [Simple magic: Synthesis and in-memory Mapping of logic execution for memristor-aided logic](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8203782) **ICCAD 2017**. Rotem Ben Hur, Nimrod Wald, Nishil Talati, Shahar Kvatinsky. _Israel Institute of Technology_
    - [SIMPLER MAGIC: Synthesis and Mapping of In-Memory Logic Executed in a single Row to Improve Throughput]() **TCAD 2020**. Rotem Ben Hur, Ronny Ronen, Ameer Haj Ali, Debjyoti Bhattacharjee, Adi Eliahu, Natan Peled, Shahar Kvatinsky. _Israel Institute of Technology_
    - [SSR: A Skeleton-based Synthesis Flow for Hybrid Processing-in-RRAM modes](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9643493) **ICCAD 2021**. Feng Wang, Guangyu Sun, Guojie Luo. _Peking University_
    - [SIMDRAM: a framework for bit-serial SIMD processing using DRAM](https://dl.acm.org/doi/pdf/10.1145/3445814.3446749) **ASPLOS 2021**. Nastaran Hajinazar, Geraldo F. Oliveira, Sven Gregorio, João Dinis Ferreira, Nika Mansouri-Ghiasi, Minesh Patel, Mohammed Alser, Saugata Ghose, Juan Gómez-Luna, Onur Mutlu. _ETH Zürich_

## Compilers for Brain-inspired Hardware
- Network Transformation and Training
    - [Bridge the Gap between Neural Networks and Neuromorphic Hardware with a Neural Network Compiler](https://dl.acm.org/doi/pdf/10.1145/3173162.3173205) **ASPLOS 2018**. 	Yu Ji, Youhui Zhang, Wenguang Chen, Yuan Xie. _Tsinghua University_
    - [NEUTRAMS: Neural network transformation and co-design under neuromorphic hardware constraints](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7783724) **MICRO 2016**. Yu Ji, Youhui Zhang, Shuangchen Li, Ping Chi, Cihang Jiang, Peng Qu, Yuan Xie, Wenguang Chen. _Tsinghua University_
- Network Mapping
    - [A Design Flow for Mapping Spiking Neural Networks to Many-Core Neuromorphic Hardware](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9643500) **ICCAD 2021**. Shihao Song, M. Lakshmi Varshika, Anup Das, Nagarajan Kandasamy. _Drexel University_
    - [Mapping spiking neural networks onto a manycore neuromorphic architecture](https://dl.acm.org/doi/pdf/10.1145/3192366.3192371) **PLDI 2018**. Chit-Kwan Lin, Andreas Wild, Gautham N. Chinya, Tsung-Han Lin, Mike Davies, Hong Wang. _INTEL_

## Compilers for SIMT GPU
- Efficient Compute-intensive Kernel Generation
    - [Roller: Fast and Efficient Tensor Compilation for Deep Learning](https://www.usenix.org/conference/osdi22/presentation/zhu) **OSDI 2022**. Hongyu Zhu, Ruofan Wu, Yijia Diao, Shanbin Ke, Haoyu Li, Chen Zhang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Wei Cui, Fan Yang, Mao Yang, and Lidong Zhou, Asaf Cidon, Gennady Pekhimenko. _University of Toronto and Microsoft Research_.
    - [Automatic Kernel Generation for Volta Tensor Cores](https://arxiv.org/abs/2006.12645) **arXiv 2020**. Somashekaracharya G. Bhaskaracharya, Julien Demouth, Vinod Grover. _NVIDIA_.
    - [Triton: an intermediate language and compiler for tiled neural network computations](https://dl.acm.org/doi/abs/10.1145/3315508.3329973) **MAPL 2019**. [code](https://github.com/openai/triton). Philippe Tillet, Hsiang-Tsung Kung, David D. Cox. _Harvard University_.
    - [Diesel: DSL for linear algebra and neural net computations on GPUs](https://dl.acm.org/doi/10.1145/3211346.3211354) **MAPL 2018**. Venmugil Elango, Norm Rubin, Mahesh Ravishankar, Hariharan Sandanagobalane, Vinod Grover. _NVIDIA_.

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
    - [Tiramisu: A Polyhedral Compiler for Expressing Fast and Portable Code](https://dl.acm.org/doi/10.5555/3314872.3314896) **CGO 2019**. [code](https://github.com/Tiramisu-Compiler/tiramisu). Riyadh Baghdadi, Jessica Ray, Malek Ben Romdhane, Emanuele Del Sozzo, Abdurrahman Akkas, Yunming Zhang, Patricia Suriana, Shoaib Kamil, Saman P. Amarasinghe. _Massachusetts Institute of Technology_.
    - [Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions](https://arxiv.org/abs/1802.04730) **arXiv 2018**. [code](https://github.com/facebookresearch/TensorComprehensions). Nicolas Vasilache, Oleksandr Zinenko, Theodoros Theodoridis, Priya Goyal, Zachary DeVito, William S. Moses, Sven Verdoolaege, Andrew Adams, Albert Cohen. _Facebook AI Research_.

- Program Synthesis
    - [EQUALITY SATURATION FOR TENSOR GRAPH SUPEROPTIMIZATION](https://arxiv.org/abs/2101.01332) **MLSys 2021**. Yichen Yang, Phitchaya Mangpo Phothilimthana, Yisu Remy Wang, Max Willsey, Sudip Roy, Jacques Pienaar. _MIT EECS & CSAIL_.
    - [Swizzle Inventor: Data Movement Synthesis for GPU Kernels](https://homes.cs.washington.edu/~emina/doc/swizzle.asplos19.pdf) **ASPLOS 2019**. Phitchaya Mangpo Phothilimthana, Archibald Samuel Elliott, An Wang, Abhinav Jangda, Bastian Hagedorn, Henrik Barthels, Samuel J. Kaufman, Vinod Grover, Emina Torlak, Rastislav Bodík. _University of California, Berkeley_.

- Compilers for Irregular Workloads
    - [FreeTensor: A Free-Form DSL with Holistic Optimizations for Irregular Tensor Programs](https://pacman.cs.tsinghua.edu.cn/~zjd/publication/pldi22-freetensor/pldi22-freetensor.pdf) **PLDI 2022**. [code](https://github.com/roastduck/FreeTensor). Shizhi Tang, Jidong Zhai, Haojie Wang, Lin Jiang, Liyan Zheng, Zhenhao Yuan, Chen Zhang. _Tsinghua University_.

- Compilers for HPC Workloads on GPU
    - [Domain-Specific Multi-Level IR Rewriting for GPU: The Open Earth Compiler for GPU-accelerated Climate Simulation](https://dl.acm.org/doi/10.1145/3469030) **TACO 2021**. [code](https://github.com/spcl/open-earth-compiler). Tobias Gysi, Christoph Müller, Oleksandr Zinenko, Stephan Herhut, Eddie Davis, Tobias Wicky, Oliver Fuhrer, Torsten Hoefler, Tobias Grosser. _ETH Zurich_.

- Distributed Optimization
    - [DISTAL: The Distributed Tensor Algebra Compiler](http://tensor-compiler.org/files/yadav-pldi22-distal.pdf) **PLDI 2022**. Rohan Yadav, Alex Aiken, and Fredrik Kjolstad. _Stanford University_.
    - [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023) **OSDI 2022**. Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Joseph E. Gonzalez, Ion Stoica. _UC Berkeley_
    - [VirtualFlow: Decoupling Deep Learning Models from the Underlying Hardware](https://proceedings.mlsys.org/paper/2022/hash/2723d092b63885e0d7c260cc007e8b9d-Abstract.html) **MLSys 2022**. Andrew Or, Haoyu Zhang, Michael None Freedman. _Princeton University_.
    - [OneFlow: Redesign the Distributed Deep Learning Framework from Scratch](https://arxiv.org/abs/2110.15032) **arXiv 2021**. Jinhui Yuan, Xinqi Li, Cheng Cheng, Juncheng Liu, Ran Guo, Shenghang Cai, Chi Yao, Fei Yang, Xiaodong Yi, Chuan Wu, Haoran Zhang, Jie Zhao. _OneFlow Research_.

## Compilers for CPU

+ Vectorization Optimization
  + [All you need is superword-level parallelism: systematic control-flow vectorization with SLP](https://dl.acm.org/doi/abs/10.1145/3519939.3523701) **PLDI 2022**. Yishen Chen, Charith Mendis, and Saman Amarasinghe. *Massachusetts Institute of Technology, USA*.
  + [VeGen: a vectorizer generator for SIMD and beyond](https://dl.acm.org/doi/abs/10.1145/3445814.3446692) **ASPLOS 2021**. Yishen Chen, Charith Mendis, Michael Carbin, and Saman Amarasinghe. *Massachusetts Institute of Technology, USA.*
  + [NeuroVectorizer: end-to-end vectorization with deep reinforcement learning](https://dl.acm.org/doi/abs/10.1145/3368826.3377928) **CGO 2020**. Ameer Haj-Ali, Nesreen K. Ahmed, Ted Willke, Yakun Sophia Shao, Krste Asanovic, and Ion Stoica. *University of California at Berkeley, USA*.
  + [Compiler Auto-Vectorization with Imitation Learning](https://proceedings.neurips.cc/paper/2019/hash/d1d5923fc822531bbfd9d87d4760914b-Abstract.html) **NIPS 2019**. Charith Mendis, Cambridge Yang, Yewen Pu, Dr.Saman Amarasinghe, Michael Carbin. *MIT CSAIL*.
  + [Translating Traditional SIMD Instructions to Vector Length Agnostic Architectures](https://ieeexplore.ieee.org/abstract/document/8661195) **CGO 2019**. Fu, Sheng-Yu and Hsu, Wei-Chung. *National Taiwan University*.
  + [Extending LLVM for Lightweight SPMD Vectorization: Using SIMD and Vector Instructions Easily from Any Language](https://ieeexplore.ieee.org/abstract/document/8661165/) **CGO 2019**. Kruppe, Robin and Oppermann, Julian and Sommer, Lukas and Koch, Andreas. *Embedded Systems and Applications Group, TU, Germany.*
  + [Super-Node SLP: Optimized Vectorization for Code Sequences Containing Operators and Their Inverse Elements](https://ieeexplore.ieee.org/abstract/document/8661192/) **CGO 2019**. V. Porpodas, R. C. O. Rocha, E. Brevnov, L. F. W. Góes and T. Mattson. *Intel Corporation, USA*.
  + [Partial control-flow linearization](https://dl.acm.org/doi/abs/10.1145/3296979.3192413) **PLDI 2018**. Moll, Simon and Hack, Sebastian. *Saarland University, Germany.*
  + [Look-ahead SLP: auto-vectorization in the presence of commutative operations](https://dl.acm.org/doi/abs/10.1145/3168807) **CGO 2018**. Vasileios Porpodas, Rodrigo C. O. Rocha, and Luís F. W. Góes. *Intel, USA*.
+ Parallelism and Locality Optimization
  + [Analytical characterization and design space exploration for optimization of CNNs](https://dl.acm.org/doi/10.1145/3445814.3446759) **ASPLOS 2021**. Rui Li, Yufan Xu, Aravind Sukumaran-Rajam, Atanas Rountev, P. Sadayappan. *University of Utah, USA*.
  + [AutoTM: Automatic Tensor Movement in Heterogeneous Memory Systems using Integer Linear Programming](https://dl.acm.org/doi/abs/10.1145/3373376.3378465) **ASPLOS 2020**. Mark Hildebrand, Jawad Khan, Sanjeev Trika, Jason Lowe-Power, and Venkatesh Akella. *University of California, Davis, Davis, CA, USA*.
  + [T4: Compiling Sequential Code for Effective Speculative Parallelization in Hardware](https://ieeexplore.ieee.org/abstract/document/9138940) **ISCA 2020**. Ying, Victor A. and Jeffrey, Mark C. and Sanchez, Daniel. *MIT CSAIL*.
  + [Optimizing data-intensive computations in existing libraries with split annotations](https://dl.acm.org/doi/abs/10.1145/3341301.3359652) **SOSP 2019**. Palkar, Shoumik and Zaharia, Matei. *Stanford University.*
  + [Model-driven transformations for multi- and many-core CPUs](https://dl.acm.org/doi/abs/10.1145/3314221.3314653) **PLDI 2019**. Kong, Martin and Pouchet, Louis-Noel. *Brookhaven National Laboratory, USA*.
+ Compilers for Sparse Workloads
  + [Efficient Execution of Graph Algorithms on CPU with SIMD Extensions](https://ieeexplore.ieee.org/abstract/document/9370326/) **CGO 2021**. Zheng, Ruohuang and Pai, Sreepathi. *Department of Computer Science, University of Rochester, USA.*
  + [Generating piecewise-regular code from irregular structures](https://dl.acm.org/doi/abs/10.1145/3314221.3314615) **PLDI 2019**. Travis Augustine, Janarthanan Sarma, Louis-Noël Pouchet, and Gabriel Rodríguez. *Colorado State University, USA.*
  + [CVR: efficient vectorization of SpMV on x86 processors](https://dl.acm.org/doi/abs/10.1145/3168818) **CGO 2018**. Biwei Xie, Jianfeng Zhan, Xu Liu, Wanling Gao, Zhen Jia, Xiwen He, and Lixin Zhang. *Institute of Computing Technology at Chinese Academy of Sciences, China*.
+ Compilers for Dense Workloads
  + [Optimizing N-dimensional, winograd-based convolution for manycore CPUs](https://dl.acm.org/doi/abs/10.1145/3178487.3178496) **PPoPP 2018**. Zhen Jia, Aleksandar Zlateski, Fredo Durand, and Kai Li. *Princeton University.*
  + [SIMD code generation for stencils on brick decompositions](https://dl.acm.org/doi/abs/10.1145/3200691.3178537) **PPoPP 2018**. Tuowen Zhao, Mary Hall, Protonu Basu, Samuel Williams, and Hans Johansen. *University of Utah.*
  + [Program generation for small-scale linear algebra applications](https://dl.acm.org/doi/abs/10.1145/3168812) **CGO 2018**. Daniele G. Spampinato, Diego Fabregat-Traver, Paolo Bientinesi, and Markus Püschel. *ETH Zurich, Switzerland*.
+ Compilers for End-to-End Networks
  + [SPNC: An Open-Source MLIR-Based Compiler for Fast Sum-Product Network Inference on CPUs and GPUs](https://ieeexplore.ieee.org/abstract/document/9741277/) **CGO 2022**. Sommer, Lukas and Axenie, Cristian and Koch, Andreas. *Embedded Systems and Applications Group, TU Darmstadt, Germany.*
  + [Multi-target Compiler for the Deployment of Machine Learning Models](https://ieeexplore.ieee.org/abstract/document/8661199) **CGO 2019**. Castro-Lopez, Oscar and Vega-Lopez, Ines F. *Facultad de Informatica, Universidad Autonoma de Sinaloa, Culiacan, Mexico.*

## Compilers for Mobile and Edge

+ Compilers for Intermittent Devices
  + [WARio: efficient code generation for intermittent computing](https://dl.acm.org/doi/abs/10.1145/3519939.3523454) **PLDI 2022**. Vito Kortbeek, Souradip Ghosh, Josiah Hester, Simone Campanoni, and Przemysław Pawełczak. *Delft University of Technology, Netherlands*.
  + [Time-sensitive Intermittent Computing Meets Legacy Software](https://dl.acm.org/doi/abs/10.1145/3373376.3378476) **ASPLOS 2020**. Vito Kortbeek, Kasim Sinan Yildirim, Abu Bakar, Jacob Sorber, Josiah Hester, and Przemysław Pawełczak. *Delft University of Technology, Delft, Netherlands*.
  + [Adaptive low-overhead scheduling for periodic and reactive intermittent execution](https://dl.acm.org/doi/abs/10.1145/3385412.3385998) **PLDI 2020**. Kiwan Maeng and Brandon Lucia. *Carnegie Mellon University, USA*.
  + [Intelligence Beyond the Edge: Inference on Intermittent Embedded Systems](https://dl.acm.org/doi/abs/10.1145/3297858.3304011) **ASPLOS 2019**. Graham Gobieski, Brandon Lucia, and Nathan Beckmann. *Carnegie Mellon University, USA*.
  + [Supporting peripherals in intermittent systems with just-in-time checkpoints](https://dl.acm.org/doi/abs/10.1145/3314221.3314613) **PLDI 2019**. Kiwan Maeng and Brandon Lucia. *Carnegie Mellon University, USA*.
+ Compilers for Digital Signal Processors
  + [GCD2: A Globally Optimizing Compiler for Mapping DNNs to Mobile DSPs](https://ieeexplore.ieee.org/document/9923837/) **MICRO 2022**. Wei Niu, Jiexiong Guan, Xipeng Shen, Yanzhi Wang, Gagan Agrawal, Bin Ren. *William & Mary, USA*.
  + [Vector instruction selection for digital signal processors using program synthesis](https://dl.acm.org/doi/abs/10.1145/3503222.3507714) **ASPLOS 2022**. Maaz Bin Safeer Ahmad, Alexander J. Root, Andrew Adams, Shoaib Kamil, and Alvin Cheung. *Adobe, USA*.
  + [Vectorization for digital signal processors via equality saturation](https://dl.acm.org/doi/abs/10.1145/3445814.3446707) **ASPLOS 2021**. Alexa VanHattum, Rachit Nigam, Vincent T. Lee, James Bornholt, and Adrian Sampson. *Cornell University, USA*.
+ Optimization for On-device Learning
  + [POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging](https://proceedings.mlr.press/v162/patil22b.html) **ICML 2022**. Shishir G. Patil, Paras Jain, Prabal Dutta, Ion Stoica, Joseph Gonzalez. *University of California Berkeley*.
  + [ZeroFL: Efficient On-Device Training for Federated Learning with Local Sparsity](https://arxiv.org/abs/2208.02507) **ICLR 2022**. Xinchi Qiu, Javier Fernandez-Marques, Pedro PB Gusmao, Yan Gao, Titouan Parcollet, Nicholas Donald Lane. *Department of Computer Science and Technology, University of Cambridge*.
  + [Distributed Distillation for On-Device Learning](https://proceedings.neurips.cc/paper/2020/hash/fef6f971605336724b5e6c0c12dc2534-Abstract.html) **NIPS 2020**. Ilai Bistritz, Ariana Mann, Nicholas Bambos. *Stanford University*.
  + [E2-Train: Training State-of-the-art CNNs with Over 80% Energy Savings](https://proceedings.neurips.cc/paper/2019/hash/663772ea088360f95bac3dc7ffb841be-Abstract.html) **NIPS 2019**. Yue Wang, Ziyu Jiang, Xiaohan Chen, Pengfei Xu, Yang Zhao, Yingyan Lin, Zhangyang Wang. *Department of Electrical and Computer Engineering, Rice University*.
+ Model Compression for Mobile Devices
  + [CoCoPIE: enabling real-time AI on off-the-shelf mobile devices via compression-compilation co-design](https://dl.acm.org/doi/10.1145/3418297?sid=SCITRUS) **Commun. ACM**. Hui Guan, Shaoshan Liu, Xiaolong Ma, Wei Niu, Bin Ren, Xipeng Shen, Yanzhi Wang, Pu Zhao. _University of Massachusetts at Amherst_.
  + [PatDNN: Achieving Real-Time DNN Execution on Mobile Devices with Pattern-based Weight Pruning](https://dl.acm.org/doi/abs/10.1145/3373376.3378534) **ASPLOS 2020**. Wei Niu, Xiaolong Ma, Sheng Lin, Shihao Wang, Xuehai Qian, Xue Lin, Yanzhi Wang, and Bin Ren. *College of William and Mary, Williamsburg, VA, USA*.
  + [Compiling KB-sized machine learning models to tiny IoT devices](https://dl.acm.org/doi/abs/10.1145/3314221.3314597) **PLDI 2019**. Sridhar Gopinath, Nikhil Ghanathe, Vivek Seshadri, and Rahul Sharma. *Microsoft Research, India*.
  + [Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking](https://openreview.net/forum?id=BylBr3C9K7) **ICLR 2019**. Haichuan Yang, Yuhao Zhu, Ji Liu. *Department of Computer Science, University of Rochester, Rochester, USA*.
+ Optimization for Mobile Device Inference
  + [Towards a Domain-Extensible Compiler: Optimizing an Image Processing Pipeline on Mobile CPUs](https://ieeexplore.ieee.org/abstract/document/9370337/) **CGO 2021**. Koehler, Thomas and Steuwer, Michel. *Philips Research, Hamburg, Germany*.
  + [AutoScale: Energy Efficiency Optimization for Stochastic Edge Inference Using Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/9251950/) **MICRO 2020**. Kim, Young Geun and Wu, Carole-Jean. *Korea University, Seoul, South Korea*.
+ Neural Architecture Search for Mobile Devices
  + [MCUNet: Tiny Deep Learning on IoT Devices](https://proceedings.neurips.cc/paper/2020/hash/86c51678350f656dcc7f490a43946ee5-Abstract.html) **NIPS 2020**. Ji Lin, Wei-Ming Chen, Yujun Lin, john cohn, Chuang Gan, Song Han. *MIT*.
  + [Constrained deep neural network architecture search for IoT devices accounting for hardware calibration](https://proceedings.neurips.cc/paper/2019/hash/f8037f94e53f17a2cc301033ca86d278-Abstract.html) **NIPS 2019**. Florian Scheidegger, Luca Benini, Costas Bekas, A. Cristiano I. Malossi. *ETH Zürich, Switzerland*.

## Compilers for RISC-V
- HLL Compilers
    - [RISCV-COLLAB](https://github.com/riscv-collab/riscv-gnu-toolchain)
    - [openarkcompiler](https://gitee.com/openarkcompiler-incubator/mapleall)
- DSL Compilers
    - [BUDDY MLIR: MLIR-Based Ideas Landing Project](https://github.com/buddy-compiler/buddy-mlir)
- Others
    - [HW/SW approaches for RISC-V code size reduction](https://carrv.github.io/2020/papers/CARRV2020_paper_12_Perotti.pdf) **CARRV 2020**. Perotti, Matteo, et al.
    - [Automatic Code Generation for Rocket Chip RoCC Accelerators](https://carrv.github.io/2020/papers/CARRV2020_paper_3_Xu.pdf) **CARRV 2020**. Xu, Pengcheng, and Yun Liang.
    - [Experiments and optimizations for TVM on RISC-V Architectures with P Extension](https://ieeexplore.ieee.org/document/9196477) **VLSI-DAT 2020**. Chen, Yi-Ru.
    - [Enabling TVM on RISC-V Architectures with SIMD Instructions](https://riscv.org/wp-content/uploads/2019/03/16.45-Enabling-TVM-on-RISC-V-Architectures-with-SIMD-Instructions-v2.pdf) **RISC-V Workshop 2019**
    - [Towards Deep Learning using TensorFlow Lite on RISC-V](https://projects.iq.harvard.edu/files/edge/files/carrv_workshop_submission_2019_camera_ready.pdf) **CARRV 2019** Louis, Marcia Sahaya, et al.


## Compilers for Configurable Hardware
  - Domain specific langage
    - [HeteroFlow: An Accelerator Programming Model with Decoupled Data Placement for Software-Defined FPGAs](https://dl.acm.org/doi/10.1145/3490422.3502369) **FPGA 2022**. Shaojie Xiang, Yihsiang Lai, Yuan Zhou, Hongzheng Chen, Niansong Zhang, Debjit Pal, and Zhiru Zhang
    - [HeteroCL: A Multi-Paradigm Programming Infrastructure for Software-Defined Reconfigurable Computing](https://dl.acm.org/doi/10.1145/3289602.3293910) **FPGA 2019**. Yihsiang Lai, Yuze Chi, Yuwei Hu, Jie Wang, Cody Hao Yu, Yuan Zhou, Jason Cong, and Zhiru Zhang
    - [T2S-Tensor: Productively Generating High-Performance Spatial Hardware for Dense Tensor Computations](https://ieeexplore.ieee.org/document/8735529) **FCCM 2019**. Nitish Kumar Srivastava, Hongbo Rong, Prithayan Barua, Guanyu Feng, Huanqi Cao, Zhiru Zhang, David H. Albonesi, Vivek Sarkar, Wenguang Chen, Paul Petersen, Geoff Lowney, Adam Herr, Christopher J. Hughes, Timothy G. Mattson and Pradeep Dubey
    - [SuSy: A Programming Model for Productive Construction of High-Performance Systolic Arrays on FPGAs](https://dl.acm.org/doi/10.1145/3400302.3415644) **ICCAD 2020**. Lai, Yi-Hsiang, Rong, Hongbo, Zheng, Size, Zhang, Weihao, Cui, Xiuping, Jia, Yunshan, Wang, Jie, Sullivan, Brendan, Zhang, Zhiru, Liang, Yun, Youhui Zhang, Jason Cong, Nithin George, Jose Alvarez, Christopher J. Hughes and Pradeep Dubey
    - [Darkroom: compiling high-level image processing code into hardware pipelines](https://dl.acm.org/doi/10.1145/2601097.2601174) **TOG 2014**. James Hegarty, John Brunhaver, Zachary DeVito, Jonathan Ragan Kelley, Noy Cohen, Steven Bell, Artem Vasilyev, Mark Horowitz and Pat Hanrahan
    - [Spatial: a language and compiler for application accelerators](https://dl.acm.org/doi/10.1145/3192366.3192379) **PLDI 2018**. David Koeplinger, Matthew Feldman, Raghu Prabhakar, Yaqi Zhang, Stefan Hadjis, Ruben Fiszel, Tian Zhao, Luigi Nardi, Ardavan Pedram, Christos Kozyrakis and Kunle Olukotun
    - [A Unified Backend for Targeting FPGAs from DSLs](https://ieeexplore.ieee.org/document/8445108) **ASAP 2018**. Emanuele Del Sozzo, Riyadh Baghdadi, Saman Amarasinghe, and Marco D. Santambrogio

## Design Space Construction and Exploration

- Graph Optimizations
    - [MapZero: Mapping for Coarse-grained Reconfigurable Architectures with Reinforcement Learning and Monte-Carlo Tree Search](https://dl.acm.org/doi/abs/10.1145/3579371.3589081) **ISCA 2023**. Xiangyu Kong, Yi Huang, Jianfeng Zhu, Xingchen Man, Yang Liu, Chunyang Feng, Pengfei Gou, Minggui Tang, Shaojun Wei, Leibo Liu. _Tsinghua_ 
    - [DeFiNES: Enabling Fast Exploration of the Depth-first Scheduling Space for DNN Accelerators through Analytical Modeling](https://arxiv.org/abs/2212.05344). **HPCA 2023**. Linyan Mei, Koen Goetschalckx, Arne Symons, Marian Verhelst. _KU Leuven_.
    - [APOLLO: AUTOMATIC PARTITION-BASED OPERATOR FUSION THROUGH LAYER BY LAYER OPTIMIZATION](https://proceedings.mlsys.org/paper/2022/hash/069059b7ef840f0c74a814ec9237b6ec-Abstract.html) **MLSys 2022**. Jie Zhao, Xiong Gao, Ruijie Xia, Zhaochuang Zhang, Deshi Chen, Lei Chen, Renwei Zhang, Zhen Geng, Bin Cheng, Xuefeng Jin. _State Key Laboratory of Mathematical Engineering and Advanced Computing_.
    - [NeoFlow: A Flexible Framework for Enabling Efficient Compilation for High Performance DNN Training](https://www.computer.org/csdl/journal/td/2022/11/09664259/1zHDLHIxWPC) **TPDS 2022**. Size Zheng, Renze Chen, Yicheng Jin, Anjiang Wei, Bingyang Wu, Xiuhong Li, Shengen Yan, Yun Liang. _Peking University_.
    - [DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion](https://dl.acm.org/doi/10.1145/3453483.3454083) **PLDI 2021**. Wei Niu, Jiexiong Guan, Yanzhi Wang, Gagan Agrawal, Bin Ren. _College of William & Mary_.
    - [DeepCuts: A Deep Learning Optimization Framework for Versatile GPU Workloads](https://dl.acm.org/doi/10.1145/3453483.3454038) **PLDI 2021** Wookeun Jung, Thanh Tuan Dao, Jaejin Lee. _Seoul National University_.
    - [Pet: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections](https://www.usenix.org/conference/osdi21/presentation/wang#:~:text=We%20propose%20PET%2C%20the%20first,only%20maintain%20partial%20functional%20equivalence.) **OSDI 2021**. [code](https://github.com/thu-pacman/PET). Haojie Wang, Jidong Zhai, Mingyu Gao, Zixuan Ma, Shizhi Tang, Liyan Zheng, Yuanzhi Li, Kaiyuan Rong, Yuanyong Chen, Zhihao Jia. _Tsinghua University_.
    - [Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks](https://www.usenix.org/conference/osdi20/presentation/ma) **OSDI 2020**. [code](https://github.com/microsoft/nnfusion). Lingxiao Ma, Zhiqiang Xie, Zhi Yang, Jilong Xue, Youshan Miao, Wei Cui, Wenxiang Hu, Fan Yang, Lintao Zhang, Lidong Zhou. _Peking University and Microsoft Research_.
    - [TASO: optimizing deep learning computation with automatic generation of graph substitutions](https://cs.stanford.edu/~padon/taso-sosp19.pdf) **SOSP 2019**. [code](https://github.com/jiazhihao/TASO). Zhihao Jia, Oded Padon, James J. Thomas, Todd Warszawski, Matei Zaharia, Alex Aiken. _Stanford University_.
    - [Relay: a new IR for machine learning frameworks](https://dl.acm.org/doi/10.1145/3211346.3211348) **MAPL 2018**. [code](https://github.com/apache/tvm/tree/main/python/tvm/relay). Jared Roesch, Steven Lyubomirsky, Logan Weber, Josh Pollock, Marisa Kirisame, Tianqi Chen, Zachary Tatlock. _University of Washington_.

- Auto-tuning and Auto-scheduling
    - [BaCO: A Fast and Portable Bayesian Compiler Optimization Framework](https://arxiv.org/pdf/2212.11142.pdf). **ASPLOS 2024**. Erik Hellsten, Artur Souza, Johannes Lenfers, Rubens Lacouture, Olivia Hsu, Adel Ejjeh, Fredrik Kjolstad, Michel Steuwer, Kunle Olukotun, and Luigi Nardi. _Stanford_
    - [Glimpse: mathematical embedding of hardware specification for neural compilation](https://cseweb.ucsd.edu/~bhahn221/doc/paper/dac22-glimpse.pdf) **DAC 2022**. Byung Hoon Ahn, Sean Kinzer, Hadi Esmaeilzadeh. _University of California_.
    - [Efficient Automatic Scheduling of Imaging and Vision Pipelines for the GPU](https://cseweb.ucsd.edu/~tzli/gpu_autoscheduler.pdf) **OOPSLA 2021**. Luke Anderson, Andrew Adams, Karima Ma, Tzu-Mao Li, Tian Jin, Jonathan Ragan-Kelley. _Massachusetts Institute of Technology_.
    - [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://www.usenix.org/system/files/osdi20-zheng.pdf) **OSDI 2020**. [code](https://github.com/apache/tvm/tree/main/python/tvm/auto_scheduler). Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, Ion Stoica. _UC Berkeley_.
    - [FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System](https://dl.acm.org/doi/10.1145/3373376.3378508) **ASPLOS 2020**. [code](https://github.com/KnowingNothing/FlexTensor). Size Zheng, Yun Liang, Shuo Wang, Renze Chen, Kaiwen Sheng. _Peking University_.
    - [ProTuner: Tuning Programs with Monte Carlo Tree Search](https://arxiv.org/abs/2005.13685) **arXiv 2020**. Ameer Haj-Ali, Hasan Genc, Qijing Huang, William S. Moses, John Wawrzynek, Krste Asanovic, Ion Stoica. _UC Berkeley_.
    - [Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation](https://cseweb.ucsd.edu/~bhahn221/doc/paper/iclr20-chameleon.pdf) **ICLR 2020**. [code](https://github.com/anony-sub/chameleon). Byung Hoon Ahn, Prannoy Pilligundla, Amir Yazdanbakhsh, Hadi Esmaeilzadeh. _University of California, San Diego_.
    - [Learning to Optimize Halide with Tree Search and Random Programs](https://dl.acm.org/doi/10.1145/3306346.3322967) **SIGGRAPH 2019**. Andrew Adams, Karima Ma, Luke Anderson, Riyadh Baghdadi, Tzu-Mao Li, Michaël Gharbi, Benoit Steiner, Steven Johnson, Kayvon Fatahalian, Frédo Durand, Jonathan Ragan-Kelley. _Facebook AI Research_.
    - [Learning to Optimize Tensor Programs](https://proceedings.neurips.cc/paper/2018/file/8b5700012be65c9da25f49408d959ca0-Paper.pdf) **NeurIPS 2018**. [code](https://github.com/apache/tvm/tree/main/python/tvm/autotvm). Tianqi Chen, Lianmin Zheng, Eddie Q. Yan, Ziheng Jiang, Thierry Moreau, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy. _University of Washington_.
    - [Automatically Scheduling Halide Image Processing Pipelines](https://dl.acm.org/doi/pdf/10.1145/2897824.2925952) **SIGGRAPH 2016**. Ravi Teja Mullapudi, Andrew Adams, Dillon Sharlet, Jonathan Ragan-Kelley, Kayvon Fatahalian. _Carnegie Mellon University_.

- Analytical Approaches
    - [Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion](https://ieeexplore.ieee.org/document/10071018). **HPCA 2023**. Size Zheng, Siyuan Chen, Peidi Song, Renze Chen, Xiuhong Li, Shengen Yan, Dahua Lin, Jingwen Leng, Yun Liang. _Peking University_.
    - [Analytical characterization and design space exploration for optimization of CNNs](https://dl.acm.org/doi/10.1145/3445814.3446759) **ASPLOS 2021**. Rui Li, Yufan Xu, Aravind Sukumaran-Rajam, Atanas Rountev, P. Sadayappan. _University of Utah_.
    - [Tuna: A Static Analysis Approach to Optimizing Deep Neural Networks](https://arxiv.org/abs/2104.14641#:~:text=Tuna%3A%20A%20Static%20Analysis%20Approach%20to%20Optimizing%20Deep%20Neural%20Networks,-Yao%20Wang%2C%20Xingyu&text=We%20introduce%20Tuna%2C%20a%20static,performance%20of%20deep%20neural%20networks.) **arXiv 2021**. Yao Wang, Xingyu Zhou, Yanming Wang, Rui Li, Yong Wu, Vin Sharma. _Amazon Web Services_.
    - [Analytical cache modeling and tilesize optimization for tensor contractions](https://hal.inria.fr/hal-02418875/document) **SC 2019**. 	Rui Li, Aravind Sukumaran-Rajam, Richard Veras, Tze Meng Low, Fabrice Rastello, Atanas Rountev, P. Sadayappan. _University of Utah_.


## Dynamic Shape and Control Flow
- Dynamic Shape Operator
    - [DietCode: Automatic Optimization for Dynamic Tensor Programs]() **MLSys 2022** [code](https://github.com/UofT-EcoSystem/DietCode) Bojian Zheng, Ziheng Jiang, Cody Hao Yu, Haichen Shen, Joshua Fromm, Yizhi Liu, Yida Wang, Luis Ceze, Tianqi Chen, Gennady Pekhimenko. *AWS*.
    - [The CoRa Tensor Compiler: Compilation for Ragged Tensors with Minimal Padding](https://proceedings.mlsys.org/paper/2022/file/d3d9446802a44259755d38e6d163e820-Paper.pdf) **MLSys 2022** Pratik Fegade, Tianqi Chen, Phillip Gibbons, Todd Mowry. _CMU_.
    - [Nimble: Efficiently Compiling Dynamic Neural Networks for Model Inference](https://proceedings.mlsys.org/paper/2021/hash/4e732ced3463d06de0ca9a15b6153677-Abstract.html) **MLSys 2021** Haichen Shen, Jared Roesch, Zhi Chen, Wei Chen, Yong Wu, Mu Li, Vin Sharma, Zachary Tatlock, Yida Wang. *AWS*.
- Dynamic Computation Graph
     - [DVABatch: Diversity-aware Multi-Entry Multi-Exit Batching for Efficient Processing of DNN Services on GPUs](https://www.usenix.org/system/files/atc22-cui.pdf)  **ATC 2022** [code](https://github.com/sjtu-epcc/DVABatch.git) Weihao Cui, Han Zhao, Quan Chen, Hao Wei, and Zirui Li,  Deze Zeng, Chao Li, Minyi Gu. _Shanghai Jiao Tong University_.
     - [Cortex: A Compiler for Recursive Deep Learning Models](https://proceedings.mlsys.org/paper/2021/file/182be0c5cdcd5072bb1864cdee4d3d6e-Paper.pdf) **MLSys 2021** Pratik Fegade, Tianqi Chen, Phillip Gibbons, Todd Mowry. _CMU_.
     - [Nimble: Efficiently Compiling Dynamic Neural Networks for Model Inference](https://proceedings.mlsys.org/paper/2021/hash/4e732ced3463d06de0ca9a15b6153677-Abstract.html) **MLSys 2021** Haichen Shen, Jared Roesch, Zhi Chen, Wei Chen, Yong Wu, Mu Li, Vin Sharma, Zachary Tatlock, Yida Wang. *AWS*.
     - [DISC: A Dynamic Shape Compiler for Machine Learning Workloads](https://dl.acm.org/doi/abs/10.1145/3437984.3458838) **EuroMLSys 2021** [code](https://github.com/alibaba/BladeDISC) K. Zhu, W.Y. Zhao, Z. Zheng, T.Y. Guo, P.Z. Zhao, J.J. Bai, J. Yang, X.Y. Liu, L.S. Diao, and W. Lin. *Alibaba*.
     - [Cavs: An Efficient Runtime System for Dynamic Neural Networks](https://www.usenix.org/system/files/conference/atc18/atc18-xu-shizhen.pdf)  **ATC 2018** [code](https://github.com/zhisbug/Cavs.git) Shizhen Xu, Hao Zhang, Graham Neubig, and Wei Dai, Jin Kyu Kim, Zhijie Deng, Qirong Ho,Guangwen Yang,Eric P. Xing. _CMU THU_.
     - [On-the-fly Operation Batching in Dynamic Computation Graphs](https://proceedings.neurips.cc/paper/2017/file/c902b497eb972281fb5b4e206db38ee6-Paper.pdf)  **NIPS17** [code](https://github.com/clab/dynet.git)  Graham Neubig, Yoav Goldberg, Chris Dyer. _CMU_.
     - [Deep Learning with Dynamic Computation Graphs](https://arxiv.org/abs/1702.02181) **ICLR 2017** Moshe Looks, Marcello Herreshoff, DeLesley Hutchins, Peter Norvig. *Google.*
     - [DyNet: The Dynamic Neural Network Toolkit](https://arxiv.org/abs/1701.03980) **arXiv 2017** [code](https://github.com/clab/dynet) Graham Neubig, Chris Dyer, Yoav Goldberg, Austin Matthews, Waleed Ammar, Antonios Anastasopoulos, Miguel Ballesteros, David Chiang, Daniel Clothiaux, Trevor Cohn, Kevin Duh, Manaal Faruqui, Cynthia Gan, Dan Garrette, Yangfeng Ji, Lingpeng Kong, Adhiguna Kuncoro, Gaurav Kumar, Chaitanya Malaviya, Paul Michel, Yusuke Oda, Matthew Richardson, Naomi Saphra, Swabha Swayamdipta, Pengcheng Yin. *CMU.*
    

## Sparse Applications, Compilers, and Architectures
- Compiler Design
    - [Mosaic: An Interoperable Compiler for Tensor Algebra](https://dl.acm.org/doi/10.1145/3591236). **PLDI 2023**. Manya Bansal, Olivia Hsu, Kunle Olukotun, Fredrik Kjolstad. _Stanford_.
    - [The Sparse Abstract Machine](https://dl.acm.org/doi/10.1145/3582016.3582051) **ASPLOS 2023**. Olivia Hsu, Maxwell Strange, Ritvik Sharma, Jaeyeon Won, Kunle Olukotun, Joel S. Emer, Mark A. Horowitz, Fredrik Kjolstad. _Stanford_.
    - [WACO: Learning Workload-Aware Co-optimization of the Format and Schedule of a Sparse Tensor Program](https://dl.acm.org/doi/10.1145/3575693.3575742) **ASPLOS 2023**. Jaeyeon Won, Charith Mendis, Joel S. Emer, Saman P. Amarasinghe. _MIT CSAIL_
    - [SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning](https://arxiv.org/pdf/2207.04606.pdf) **ASPLOS 2023**. Zihao Ye, Ruihang Lai, Junru Shao, Tianqi Chen, Luis Ceze. _University of Washington_.
    - [Looplets: A Language for Structured Coiteration](https://dl.acm.org/doi/10.1145/3579990.3580020) **CGO 2023** Willow Ahrens, Daniel Donenfeld, Fredrik Kjolstad, Samam P. Amarasinghe. _MIT CSAIL_
    - [Autoscheduling for Sparse Tensor Algebra with an Asymptotic Cost Model](http://tensor-compiler.org/files/ahrens-pldi22-taco-autoscheduling.pdf) **PLDI 2022**. Peter Ahrens, Fredrik Kjolstad, and Saman Amarasinghe. _MIT CSAIL_.
    - [Unified Compilation for Lossless Compression and Sparse Computing](https://groups.csail.mit.edu/commit/papers/2022/cgo22-TACO-lossless.pdf) **CGO 2022**. Daniel Donenfeld, Stephen Chou, and Saman Amarasinghe. _MIT CSAIL_.
    - [Dynamic Sparse Tensor Algebra Compilation](https://arxiv.org/abs/2112.01394) **arXiv 2021**. Stephen Chou and Saman Amarasinghe. _MIT CSAIL_.
    - [Compilation of Sparse Array Programming Models](https://dl.acm.org/doi/abs/10.1145/3485505) **OOPSLA 2021**. Rawn Henry, Olivia Hsu, Rohan Yadav, Stephen Chou, Kunle Olukotun, Saman Amarasinghe, and Fredrik Kjolstad. _MIT CSAIL_.
    - [A sparse iteration space transformation framework for sparse tensor algebra](https://dl.acm.org/doi/10.1145/3428226) **OOPSLA 2020**. Ryan Senanayake, Changwan Hong, Ziheng Wang, Amalee Wilson, Stephen Chou, Shoaib Kamil, Saman P. Amarasinghe, Fredrik Kjolstad. _Reservoir Labs_.
    - [Automatic Generation of Efficient Sparse Tensor Format Conversion Routines](https://arxiv.org/pdf/2001.02609.pdf) **PLDI 2020**. Stephen Chou, Fredrik Kjolstad, and Saman Amarasinghe. _MIT CSAIL_.
    - [Tensor Algebra Compilation with Workspaces](http://fredrikbk.com/publications/taco-workspaces.pdf) **CGO 2019**. Fredrik Kjolstad, Peter Ahrens, Shoaib Kamil, Saman P. Amarasinghe. _MIT_.

## Tree and Graph Applications, Compilers, and Architectures
- Compiler Design
    - [Treebeard: An Optimizing Compiler for Decision Tree Based ML Inference]() **MICRO 2022**. Ashwin Prasad, Sampath Rajendra, Kaushik Rajan, R Govindarajan, Uday Bondhugula. _Indian Institute of Science-Bangalore_.
    - [GraphIt to CUDA Compiler in 2021 LOC: A Case for High-Performance DSL Implementation via Staging with BuilDSL](https://intimeand.space/docs/CGO2022-BuilDSL.pdf) **CGO 2022**. Ajay Brahmakshatriya, Saman P. Amarasinghe. _CSAIL, MIT_.
    - [Taming the Zoo: The Unified GraphIt Compiler Framework for Novel Architectures](https://ieeexplore.ieee.org/document/9499863) **ISCA 2021**. Ajay Brahmakshatriya, Emily Furst, Victor A. Ying, Claire Hsu, Changwan Hong, Max Ruttenberg, Yunming Zhang, Dai Cheol Jung, Dustin Richmond, Michael B. Taylor, Julian Shun, Mark Oskin, Daniel Sánchez, Saman P. Amarasinghe. _MIT CSAIL_.
    - [A Tensor Compiler for Unified Machine Learning Prediction Serving](https://www.usenix.org/conference/osdi20/presentation/nakandala) **OSDI 2020**. [code](https://github.com/microsoft/hummingbird). Supun Nakandala, Karla Saur, Gyeong-In Yu, Konstantinos Karanasos, Carlo Curino, Markus Weimer, Matteo Interlandi. _UC San Diego_.
    - [Optimizing ordered graph algorithms with GraphIt](https://arxiv.org/abs/1911.07260) **CGO 2020**. Yunming Zhang, Ajay Brahmakshatriya, Xinyi Chen, Laxman Dhulipala, Shoaib Kamil, Saman P. Amarasinghe, Julian Shun. _MIT CSAIL_.
    - [GraphIt: A High-Performance Graph DSL](https://dl.acm.org/doi/10.1145/3276491) **OOPSLA 2018**. Yunming Zhang, Mengjiao Yang, Riyadh Baghdadi, Shoaib Kamil, Julian Shun, Saman P. Amarasinghe. _MIT CSAIL_.


## NAS Compilers and Architectures
- Compiler Design
    - [Neural Architecture Search as Program Transformation Exploration](https://dl.acm.org/doi/abs/10.1145/3445814.3446753) **ASPLOS 2021**. Jack Turner, Elliot J. Crowley, Michael F. P. O'Boyle. _University of Edinburgh United Kingdom_.
    
- Architecture Design
    - [NASA: Accelerating Neural Network Design with a NAS Processor](https://ieeexplore.ieee.org/document/9499756) **ISCA 2021**. Xiaohan Ma, Chang Si, Ying Wang, Cheng Liu, Lei Zhang. _CAS University of Chinese Academy of Sciences_.


## Security and Privacy
- Compiler Design
    - [PlaidML-HE: Acceleration of Deep Learning Kernels to Compute on Encrypted Data](https://ieeexplore.ieee.org/document/8988676) **ICCD 2019**. Huili Chen, Rosario Cammarota, Felipe Valencia, Francesco Regazzoni. _Intel AI Privacy and Security Research_.




## Cost Model
- Model Design
    - [TLP: A Deep Learning-Based Cost Model for Tensor Program Tuning](https://dl.acm.org/doi/10.1145/3575693.3575737). **ASPLOS 2023**. Yi Zhai, Yu Zhang, Shou Liu, Xiaomeng Chu, Jie Peng, Jianmin Ji, Yayong Zhang. _University of Science and Technology of China_.
    - [A LEARNED PERFORMANCE MODEL FOR TENSOR PROCESSING UNITS](https://arxiv.org/abs/2008.01040) **MLSys 2021**. Samuel J. Kaufman, Phitchaya Mangpo Phothilimthana, Yanqi Zhou, Charith Mendis, Sudip Roy, Amit Sabne, Mike Burrows. _Google_.

- Dataset
    - [TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/a684eceee76fc522773286a895bc8436-Paper-round1.pdf) **NeurIPS Datasets and Benchmarks 2021**. [code](https://github.com/tlc-pack/tenset). Lianmin Zheng, Ruochen Liu, Junru Shao, Tianqi Chen, Joseph Gonzalez, Ion Stoica, Ameer Haj-Ali. _UC Berkeley_.


## Hardware Design and Validation
- Synthesis
    - [PDL: a high-level hardware design language for pipelined processors](https://dl.acm.org/doi/abs/10.1145/3519939.3523455). **PLDI 2022**. Drew Zagieboylo, Charles Sherk, Gookwon Edward Suh, and Andrew C. Myers. _Cornell University_
    - [HECTOR: A Multi-level Intermediate Representation for Hardware Synthesis Methodologies](https://ieeexplore.ieee.org/document/10068908). **ICCAD 2022**. [code](https://github.com/pku-liang/Hector). Ruifan Xu, Youwei Xiao, and Yun Liang. _Peking University_
    - [A compiler infrastructure for accelerator generators](https://dl.acm.org/doi/abs/10.1145/3445814.3446712). **ASPLOS 2021**. [code](https://github.com/cucapra/calyx/). Rachit Nigam, Samuel Thomas, Zhijing Li, and Adrian Sampson. _Cornell University_
    - [LLHD: a multi-level intermediate representation for hardware description languages](https://dl.acm.org/doi/abs/10.1145/3385412.3386024). **PLDI 2020**. Fabian Schuiki, Andreas Kurth, Tobias Grosser, and Luca Benini. _ETH Zurich_
    - [Reusability is FIRRTL ground: hardware construction languages, compiler frameworks, and transformations](https://dl.acm.org/doi/10.5555/3199700.3199728). **ICCAD 2017**. Adam Izraelevitz, Jack Koenig, Patrick Li, Richard Lin, Angie Wang, Albert Magyar, Donggyu Kim, Colin Schmidt, Chick Markley, Jim Lawson, and Jonathan Bachrach. _UC Berkeley_
- Debugging
    - [Stepwise Debugging for Hardware Accelerators](https://dl.acm.org/doi/10.1145/3575693.3575717). **ASPLOS 2023**. Griffin Berlstein, Rachit Nigam, Christophe Gyurgyik, and Adrian Sampson. _Cornell University_
    - [Debugging in the brave new world of reconfigurable hardware](https://dl.acm.org/doi/10.1145/3503222.3507701). **ASPLOS 2022**. Jiacheng Ma, Gefei Zuo, Kevin Loughlin, Haoyang Zhang, Andrew Quinn, and Baris Kasikci. _University of Michigan_
    - [CirFix: Automatically Repairing Defects in Hardware Design Code](https://dl.acm.org/doi/10.1145/3503222.3507763). **ASPLOS 2022**. Hammad Ahmad, Yu Huang, and Westley Weimer. _University of Michigan_
- Simulation
    - [RepCut: Superlinear Parallel RTL Simulation with Replication-Aided Partitioning](https://dl.acm.org/doi/10.1145/3582016.3582034). **ASPLOS 2023**. Haoyuan Wang and Scott Beamer. _University of California at Santa Cruz_
## Survey and Books

## Talks, Tutorials, and Videos
