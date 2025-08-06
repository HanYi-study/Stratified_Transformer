# Stratified_Transformer  

## 环境配置：
```   
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
   - ``` (base) hy@hy:~/projects$ conda activate stratified_transformer
    (stratified_transformer) hy@hy:~/projects$ ls -l /dev/nvidia*  
    crw-rw-rw- 1 root   root    195,   0 Aug  5 09:19 /dev/nvidia0  
    crw-rw-rw- 1 root   root    195,   1 Aug  5 09:19 /dev/nvidia1  
    crw-rw-rw- 1 nobody nogroup 195, 255 Aug  5 09:00 /dev/nvidiactl ``` 
    - /dev/nvidiactl 的属主是 nobody:nogroup，这是 不正常的，可能会导致 PyTorch 无法正确初始化 CUDA。
    - 所有设备都开放了 crw-rw-rw- 权限（即 world readable/writable），虽然看似可以访问，但如果 nvidiactl 归属不对，PyTorch 初始化仍然会失败。
 - sudo usermod -aG video $USER之后，又重新sudo reboot
 - ``` (stratified_transformer) hy@hy:~/projects$ ls -l /dev/nvidia*   
    crw-rw-rw- 1 nobody nogroup 504,   0 Aug  5 09:25 /dev/nvidia-uvm  
    crw-rw-rw- 1 nobody nogroup 504,   1 Aug  5 09:25 /dev/nvidia-uvm-tools  
    crw-rw-rw- 1 root   root    195,   0 Aug  6 01:10 /dev/nvidia0  
    crw-rw-rw- 1 root   root    195,   1 Aug  6 01:10 /dev/nvidia1  
    crw-rw-rw- 1 nobody nogroup 195, 255 Aug  5 09:00 /dev/nvidiactl  # 未成功解决权限问题 ```
---
- ``` (stratified_transformer) hy@hy:~/projects$ python   
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
```
conda create -n stratified_transformer python=3.8  # open3d==0.10.0.0 只支持 Python >=3.6 且 <3.9，你的Python是3.9所以不符合要求
conda activate stratified_transformer  
pip install --upgrade pip setuptools wheel  
# 修改requirements.txt中setuptools==50.3.1.post20201107为setuptools>=50.3.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116

```
