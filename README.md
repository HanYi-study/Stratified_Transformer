# Stratified_Transformer  

## 环境配置：
- 第一次安装（失败）：
``` bash  
conda create -n stratified_transformer python=3.8
conda activate stratified_transformer
wget https://download.pytorch.org/whl/cu116/torch-1.12.1%2Bcu116-cp38-cp38-linux_x86_64.whl -O torch-1.12.1+cu116-cp38-cp38-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu116/torchvision-0.13.1%2Bcu116-cp38-cp38-linux_x86_64.whl -O torchvision-0.13.1+cu116-cp38-cp38-linux_x86_64.whl
pip install torch-1.12.1+cu116-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.13.1+cu116-cp38-cp38-linux_x86_64.whl
#使用pip安装两个包（确保在虚拟环境中）
```

>出现的问题：  
   - ``` (stratified_transformer) hy@hy:~/projects$ python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"  
     1.12.1+cu116  
     /home/hy/miniconda3/envs/stratified_transformer/lib/python3.8/site-packages/torch/cuda/__init__.py:83: UserWarning: CUDA initialization: CUDA unknown error - this   may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero.     (Triggered internally at  ../c10/cuda/CUDAFunctions.cpp:109.)  
     return torch._C._cuda_getDeviceCount() > 0  
     False  
>问题排查：
 - nvidia-smi 正常，说明驱动无误、GPU 可用  
 - 清除错误的环境变量配置（如屏蔽了 GPU）：运行：'''echo $CUDA_VISIBLE_DEVICES'''，如果输出为空字符串（""），说明你手动屏蔽了 GPU，运行'''unset CUDA_VISIBLE_DEVICES'''取消屏蔽  
 - 确认驱动版本是否兼容使用的是 PyTorch 1.12.1 + CUDA 11.6，要求系统驱动 ≥ 510.x，推荐版本为 510.47.03 或更高。在 nvidia-smi 输出中确认 Driver Version 是否满足要求（满足）  
 - reboot之后  
 - 再次：python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"  
 - 还是有问题（还是同第一次一个结果）
 - 检查 NVIDIA 驱动是否以用户模式正确加载：
   - ```
     (base) hy@hy:~/projects$ conda activate stratified_transformer  
     (stratified_transformer) hy@hy:~/projects$ ls -l /dev/nvidia*  
     crw-rw-rw- 1 root   root    195,   0 Aug  5 09:19 /dev/nvidia0  
     crw-rw-rw- 1 root   root    195,   1 Aug  5 09:19 /dev/nvidia1  
     crw-rw-rw- 1 nobody nogroup 195, 255 Aug  5 09:00 /dev/nvidiactl  
    - /dev/nvidiactl 的属主是 nobody:nogroup，这是 不正常的，可能会导致 PyTorch 无法正确初始化 CUDA。
    - 所有设备都开放了 crw-rw-rw- 权限（即 world readable/writable），虽然看似可以访问，但如果 nvidiactl 归属不对，PyTorch 初始化仍然会失败。  
 -``` sudo usermod -aG video $USER```之后，又重新sudo reboot
 - ``` (stratified_transformer) hy@hy:~/projects$ ls -l /dev/nvidia*   
    crw-rw-rw- 1 nobody nogroup 504,   0 Aug  5 09:25 /dev/nvidia-uvm  
    crw-rw-rw- 1 nobody nogroup 504,   1 Aug  5 09:25 /dev/nvidia-uvm-tools  
    crw-rw-rw- 1 root   root    195,   0 Aug  6 01:10 /dev/nvidia0  
    crw-rw-rw- 1 root   root    195,   1 Aug  6 01:10 /dev/nvidia1  
    crw-rw-rw- 1 nobody nogroup 195, 255 Aug  5 09:00 /dev/nvidiactl  # 未成功解决权限问题 
    (stratified_transformer) hy@hy:~/projects$ python   
    Python 3.8.20 (default, Oct  3 2024, 15:24:27)   
    [GCC 11.2.0] :: Anaconda, Inc. on linux  
    Type "help", "copyright", "credits" or "license" for more information.  
    >>> import torch  
    t(torch.cuda.get_device_name(0))  
    >>> print(torch.cuda.is_available())  
    True  
    >>> print(torch.cuda.get_device_name(0))  
    NVIDIA GeForce RTX 4090 D ```
- 当前环境配置确认：  
  Python 版本：3.8.20  
  CUDA 驱动版本：11.6  
  PyTorch GPU 支持： 已启用 (torch.cuda.is_available() = True)  
  GPU 型号：NVIDIA GeForce RTX 4090 D

---
- 第二次安装（失败）：
```bash
conda create -n stratified_transformer python=3.8  # open3d==0.10.0.0 只支持 Python >=3.6 且 <3.9，你的Python是3.9所以不符合要求
conda activate stratified_transformer  
pip install --upgrade pip setuptools wheel  
# 修改requirements.txt中setuptools==50.3.1.post20201107为setuptools>=50.3.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
# 运行结果从出现如下ERROR：
#ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
#torchaudio 0.13.1+cu116 requires torch==1.13.1, but you have torch 1.7.1 which is incompatible.
#可以发现：你系统中已经安装了 torchaudio 0.13.1+cu116，它要求 torch==1.13.1，但你当前安装的是 torch==1.7.1。这个问题不会影响当前大部分环境运行，尤其是如果你的项目（比如 stratified_transformer）没有用到 torchaudio。如果你的代码中不涉及音频处理模块（如 torchaudio.load() 等），你可以忽略这个冲突。
# 如果你只是运行 3D 点云模型（如 torch_points3d, torch_geometric），并且你不需要音频相关功能，那么可以不用处理这个报错，环境算是配置成功的

#现在遇到问题：刚才通过 requirements.txt 安装依赖时引入的PyTorch 1.7.1 + CUDA 10.2版本太低，不支持电脑上的 RTX 4090 D (sm_89) 显卡架构
#应该：要在不新建环境的前提下继续使用 RTX 4090 编译 pointops2，你必须手动升级 PyTorch 和 CUDA 到兼容版本
pip uninstall torch torchvision torchaudio -y  #卸载当前旧版本 PyTorch
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118  # 安装支持 sm_89 的 PyTorch（CUDA 11.8，适配 RTX 4090）

```
---
- 第三次安装（成功）：
```bash
conda create -n stratified_transformer_02 python=3.7
conda activate stratified_transformer_02
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
# 上面这两行是安装cuda11.8（在安装11.8之前我把所有的cuda全部删除了，后期还得再下回来，删除的原因是我没搞懂cuda存在于哪里，由谁使用）
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# 安装好cuda之后必须配置好环境变量
source ~/.bashrc
# 保存配置好的环境变量
nvcc --version
# 检查cuda版本是否正确
ls -l /usr/local/cuda
# 查看软链接是否指向cuda11.8
pip install torch_sparse==0.6.12
pip install torch_points3d==1.3.0
# 桉顺序下载这两个
pip install timm==0.4.9
pip install tensorboard
pip install termcolor
pip install tensorboard

```
运行成功：
```bash
Installed /home/hy/miniconda3/envs/stratified_transformer_02/lib/python3.7/site-packages/pointops2-0.0.0-py3.7-linux-x86_64.egg
Processing dependencies for pointops2==0.0.0
Finished processing dependencies for pointops2==0.0.0
```
 - --------------------------------------------------------------------------------------------------------------------------finished 25/08/06 20：42
---
## S3DIS数据集准备（数据集预处理）
**准备预处理需要的文件/文件夹，并执行预处理可执行文件**  
预处理参考如下帖子：https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master/data_utils  
1. 创建data_prepare文件夹（存储S3DIS数据预处理的可执行文件），创建data_prepare_output（存储预处理结果）
2. 在data_prepare文件夹中创建文件/文件夹，目录结构如下所示：
```text
data_prepare/
├── __pycache__/
│   └── indoor3d_util.cpython-37.pyc
├── meta/
│   ├── anno_paths.txt
│   └── class_names.txt
├── collect_indoor3d_data.py
└── indoor3d_util.py
```
3. 在**collect_indoor3d_data.py**中修改路径  
- 待处理数据的文件路径**DATA_PATH**为：/home/hy/projects/data/data_S3DIS/Stanford3dDataset_v1.2_Aligned_Version
- 预处理后存放结果的文件路径**output_folder**为：/home/hy/projects/Stratified_Transformer/data_prepare_output
ps:直接定义了DATA_PATH，更加可靠方便
4. 执行数据预处理文件**collect_indooe3d_data.py**
  ```bash
  python collect_indooe3d_data.py
  ```
  
- 生成了.txt文件  如：/home/hy/projects/data/data_S3DIS/Stanford3dDataset_v1.2_Aligned_Version/Area_6/pantry_1/Annotations/wall_4.txt  
/home/hy/projects/data/data_S3DIS/Stanford3dDataset_v1.2_Aligned_Version/Area_6/pantry_1/Annotations/clutter_5.txt  
/home/hy/projects/data/data_S3DIS/Stanford3dDataset_v1.2_Aligned_Version/Area_6/pantry_1/Annotations/ceiling_1.txt  
/home/hy/projects/data/data_S3DIS/Stanford3dDataset_v1.2_Aligned_Version/Area_6/pantry_1/Annotations/clutter_10.txt  
/home/hy/projects/data/data_S3DIS/Stanford3dDataset_v1.2_Aligned_Version/Area_6/pantry_1/Annotations/clutter_2.txt  
...
 - -----------------------------------------------------------------------------------------------finished 25/08/07 16：22
  ---
  ## 运行训练模型train.py
```bash
conda activate stratifier_transformer_02
cd /home/hy/projects/Stratified_Transformer
pip install h5py  # 报错缺少h5py包
conda install -c conda-forge sharedarray   # 报错缺少sharedarray包
# 建议直接用 conda-forge 安装，省得折腾 gcc 配置，而且和你环境的 numpy 版本也会自动匹配。conda-forge 里有预编译的 SharedArray，不需要本地编译器
conda install -c conda-forge cudatoolkit=11.1    
# 在运行train.py时报错PyTorch Geometric + CUDA 库版本不匹配
# PyTorch 是 11.1，我们直接在 stratified_transformer_02 环境里补上 11.1 的 CUDA 运行时库
# 这样会把 libcusparse.so.11、libcublas.so.11 等 CUDA 运行时库放到当前环境里，PyTorch 和 torch_sparse、torch_geometric 在加载时就能找到它们。
pip uninstall torch-cluster torch-scatter torch-sparse torch-geometric torch-spline-conv -y # 先卸载掉原先的包
pip install ./torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl
pip install ./torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
pip install ./torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install ./torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install ./torch_geometric-1.7.2-py3-none-any.whl
# 用官方whl包手动安装，访问 https://data.pyg.org/whl/torch-1.8.0+cu111.html（针对 Linux + Python 3.7，选择了四个以上文件，先下载到本地，再移动到项目文件中），再在项目中pip
# PyTorch Geometric 生态里的某个库（torch_cluster）跟你现在的 Python 版本太旧，不兼容。
# 现在仍想在3.7中配置，pip完上面四个文件之后，会显示一部分error，但仍是successfully，但仍存在依赖冲突，需要安装numpy=1.195.
pip install numpy==1.19.5 --force-reinstall
# 虽然conda list中显示numpy=1.19.5
# 但是python -c "import numpy; print(numpy.__version__)"结果是numpy=1.21.6
# 所以需要强制重新安装指定numpy版本
# 再python -c "import numpy; print(numpy.__version__)"时会输出numpy=1.19.5
