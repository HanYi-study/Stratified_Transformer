# Stratified_Transformer  

## 环境配置：
'''    
conda create -n stratified_transformer python=3.8
conda activate stratified_transformer
wget https://download.pytorch.org/whl/cu116/torch-1.12.1%2Bcu116-cp38-cp38-linux_x86_64.whl -O torch-1.12.1+cu116-cp38-cp38-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu116/torchvision-0.13.1%2Bcu116-cp38-cp38-linux_x86_64.whl -O torchvision-0.13.1+cu116-cp38-cp38-linux_x86_64.whl
pip install torch-1.12.1+cu116-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.13.1+cu116-cp38-cp38-linux_x86_64.whl
#使用pip安装两个包（确保在虚拟环境中）
>出现的问题：
(stratified_transformer) hy@hy:~/projects$ python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
1.12.1+cu116
/home/hy/miniconda3/envs/stratified_transformer/lib/python3.8/site-packages/torch/cuda/__init__.py:83: UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at  ../c10/cuda/CUDAFunctions.cpp:109.)
  return torch._C._cuda_getDeviceCount() > 0
False
