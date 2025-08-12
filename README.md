# Stratified_transformer模型项目文件夹
## 训练结果
- 训练速度：平均5.5min一个epoch，batch_size=16（修改前为8，速度太慢）
- best IoU = 7.7%
- mIoU ≈ 7%

---

## 本仓库文件介绍
- Stratified_Transformer_instructions.md：介绍整个训练过程，环境配置-->数据集准备-->执行训练脚本-->训练记录/结果
- Stratified_Transformer_QandA.md：记录Stratified Transformer环境搭建与训练遇到的问题

---

## 文件目录结构
```text
Stratified_Transformer/
├── .gitignore                        # Git忽略文件配置
├── build_log.txt                     # 构建或安装日志
├── convert_to_official_format.py     # 转换数据为官方格式的脚本
├── cuda_11.1.0_455.23.05_linux.run   # CUDA 11.1 安装包
├── cuda_11.6.0_510.39.01_linux.run   # CUDA 11.6 安装包
├── LICENSE.md                        # 许可证文件
├── README.md                         # 项目说明文档
├── requirements.txt                  # Python依赖包列表
├── test.py                           # 测试脚本
├── torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl      # 依赖whl包
├── torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl      # 依赖whl包
├── torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl       # 依赖whl包
├── torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl  # 依赖whl包
├── train.py                            # 训练主程序
│
├── __pycache__/                        # Python编译缓存
│   └── indoor3d_util.cpython-37.pyc
│
├── config/                             # 配置文件目录
│   ├── s3dis/
│   │    ├── s3dis_stratified_transformer.yaml        # S3DIS数据集的Stratified Transformer配置
│   │    └── s3dis_swin3d_transformer.yaml            # S3DIS数据集的Swin3D Transformer配置
│   └── scannetv2/
│        ├── scannetv2_stratified_transformer.yaml    # ScanNetV2数据集的Stratified Transformer配置
│        └── scannetv2_swin3d_transformer.yaml        # ScanNetV2数据集的Swin3D Transformer配置
│
├── data_prepare/                       # 数据预处理脚本目录
│   ├── collect_indoor3d_data.py        # S3DIS数据收集脚本
│   ├── indoor3d_util.py                # S3DIS数据处理工具
│   ├── __pycache__/
│   │   └── indoor3d_util.cpython-37.pyc
│   └── meta/
│
├── data_prepare_output/                # S3DIS预处理后npy数据
│   ├── Area_1_conferenceRoom_1.npy
│   ├── Area_1_conferenceRoom_2.npy
│   ├── Area_1_copyRoom_1.npy
│   ├── ...                             # 其他房间npy文件
│   └── Area_2_office_11.npy
│   └── ...（大量npy文件，按房间命名）
│
├── data_prepare_output_converted/      # 转换为官方格式后的数据
│   ├── Area_1_conferenceRoom_1.npy
│   ├── Area_1_conferenceRoom_2.npy
│   ├── Area_1_copyRoom_1.npy
│   ├── ...                             # 其他房间npy文件
│   └── Area_2_office_11.npy
│   └── ...（大量npy文件，按房间命名）
│
├── figs/                               # 可视化图片
│   └── fig.jpg
│
├── lib/                                # C++/CUDA 加速算子与底层库
│   ├── cpp_wrappers/                   # C++/CUDA 点云下采样与工具库
│   │   ├── compile_wrappers.sh         # 编译所有 C++/CUDA 封装脚本
│   │   ├── cpp_subsampling/            # 点云下采样相关 C++ 实现
│   │   │   ├── setup.py
│   │   │   ├── wrapper.cpp
│   │   │   └── grid_subsampling/
│   │   │       ├── grid_subsampling.cpp
│   │   │       └── grid_subsampling.h
│   │   └── cpp_utils/                  # 通用 C++ 工具
│   │       ├── cloud/
│   │       │   ├── cloud.cpp
│   │       │   └── cloud.h
│   │       └── nanoflann/
│   │           └── nanoflann.hpp
│   ├── pointops/                       # 点云操作 CUDA 扩展（第一代）
│   │   ├── __init__.py
│   │   ├── setup.py
│   │   ├── functions/
│   │   │   ├── __init__.py
│   │   │   └── pointops.py
│   │   └── src/
│   │       ├── __init__.py
│   │       ├── cuda_utils.h
│   │       ├── pointops_api.cpp
│   │       ├── ballquery/
│   │       │   ├── ballquery_cuda_kernel.cu
│   │       │   ├── ballquery_cuda_kernel.h
│   │       │   └── ballquery_cuda.cpp
│   │       ├── featured...
│   │       ├── grouping/
│   │       │   ├── grouping_cuda_kernel.cu
│   │       │   ├── grouping_cuda_kernel.h
│   │       │   └── grouping_cuda.cpp
│   │       ├── grouping_int/
│   │       │   ├── grouping_int_cuda_kernel.cu
│   │       │   ├── grouping_int_cuda_kernel.h
│   │       │   └── grouping_int_cuda.cpp
│   │       ├── interpolation/
│   │       │   ├── interpolation_cuda_kernel.cu
│   │       │   ├── interpolation_cuda_kernel.h
│   │       │   └── interpolation_cuda.cpp
│   │       ├── knnquery/
│   │       │   ├── __init__.py
│   │       │   ├── knnquery_cuda_kernel.cu
│   │       │   ├── knnquery_cuda_kernel.h
│   │       │   └── knnquery_cuda.cpp
│   │       ├── knnquery_heap/
│   │       │   ├── __init__.py
│   │       │   ├── knnquery_heap_cuda_kernel.cu
│   │       │   ├── knnquery_heap_cuda_kernel.h
│   │       │   └── ...
│   │       ├── labelstat/
│   │       └── sampling/
│   └── pointops2/                      # 点云操作 CUDA 扩展（第二代）
│       ├── __init__.py
│       ├── build_log.txt
│       ├── cuda_11.8.0_520.61.05_linux.run
│       ├── cuda_11.8.0_520.61.05_linux.run.1
│       ├── setup.py
│       ├── __pycache__/
│       │   └── __init__.cpython-37.pyc
│       ├── build/
│       │   ├── bdist.linux-x86_64/
│       │   ├── lib.linux-x86_64-cpython-37/
│       │   └── temp.linux-x86_64-cpython-37/
│       ├── functions/
│       │   ├── __init__.py
│       │   ├── pointops_ablation.py
│       │   ├── pointops.py
│       │   ├── pointops2.py
│       │   ├── test_attention_op_step1_v2.py
│       │   ├── test_attention_op_step1.py
│       │   ├── test_attention_op_step2.py
│       │   ├── test_relative_pos_encoding_op_step1_v2.py
│       │   ├── test_relative_pos_encoding_op_step1_v3.py
│       │   ├── test_relative_pos_encoding_op_step1.py
│       │   ├── test_relative_pos_encoding_op_step2_v2.py
│       │   ├── test_relative_pos_encoding_op_step2.py
│       │   └── __pycache__/
│       ├── pointops2.egg-info/
│       │   ├── dependency_links.txt
│       │   ├── PKG-INFO
│       │   ├── SOURCES.txt
│       │   └── top_level.txt
│       └── src/
│           ├── __init__.py
│           ├── cuda_utils.h
│           ├── pointops_api.cpp
│           ├── aggregation/
│           ├── attention/
│           ├── attention_v2/
│           ├── grouping/
│           ├── interpolation/
│           ├── knnquery/
│           ├── rpe/
│           ├── rpe_v2/
│           ├── sampling/
│           └── subtraction/
│    
│
├── model/                              # 训练得到的模型参数等
│   ├── stratified_transformer.py       # Stratified Transformer 主模型结构定义
│   ├── swin3d_transformer.py           # Swin3D Transformer 主模型结构定义
│   └── __pycache__/                    # Python 编译缓存
│       └── stratified_transformer.cpython-37.pyc
│
├── runs/                               # 训练/验证日志与模型保存
│   └── s3dis_stratified_transformer/
│       ├── events.out.tfevents.*       # tensorboard日志
│       ├── log.txt                     # 训练日志
│       └── model/
│           ├── model_best.pth          # 最优模型权重
│           └── model_last.pth          # 最新模型权重
│
├── util/                          # 工具函数与数据集相关代码
    ├── __init__.py                # 包初始化文件
    ├── common_util.py             # 通用工具函数
    ├── config.py                  # 配置文件解析与管理
    ├── data_util.py               # 数据处理相关工具
    ├── dataset.py                 # 数据集基类与加载逻辑
    ├── logger.py                  # 日志工具
    ├── lr.py                      # 学习率调度相关
    ├── s3dis.py                   # S3DIS数据集相关处理与Dataset类
    ├── scannet_v2.py              # ScanNetV2数据集相关处理与Dataset类
    ├── transform.py               # 数据增强与变换
    ├── vis_util.py                # 可视化工具
    ├── voxelize.py                # 体素化相关工具
    └── __pycache__/               # Python编译缓存
        ├── __init__.cpython-37.pyc
        ├── common_util.cpython-37.pyc
        ├── config.cpython-37.pyc
        ├── data_util.cpython-37.pyc
        ├── dataset.cpython-37.pyc
        ├── logger.cpython-37.pyc
        ├── lr.cpython-37.pyc
        ├── s3dis.cpython-37.pyc
        ├── scannet_v2.cpython-37.pyc
        ├── transform.cpython-37.pyc
        └── voxelize.cpython-37.pyc
