# Stratified Transformer 环境搭建与训练问题全记录

## 1. CUDA 扩展与 PyTorch 兼容性问题

### 问题描述

- 训练时报错 `Segmentation fault` 或 gdb 回溯指向 `torch_sparse/_version_cuda.so`。
- pip 安装时提示依赖冲突或版本不兼容。

### 原因

- PyTorch 及其相关 CUDA 扩展（如 torch-scatter、torch-sparse、torch-cluster、torch-spline-conv、torch-geometric）**必须严格版本对应**，否则极易出现段错误或符号找不到。
- 有些包（如 torch-points-kernels）需要和当前 PyTorch ABI 完全一致，不能直接用旧的 wheel。

### 解决办法

#### 1.1 卸载所有相关扩展

```sh
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric torch-points-kernels -y
```

#### 1.2 按 PyTorch 版本重新安装官方 wheel

以 PyTorch 1.13.1+cu116 为例：

```sh
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-geometric==1.7.2
```

> **注意**：torch-points3d 1.3.0 只兼容 torch-geometric <2.0.0，所以要指定 1.7.2。

#### 1.3 源码编译 torch-points-kernels

```sh
pip install torch-points-kernels==0.6.9 --no-binary :all:
```

---

## 2. GLIBCXX 版本不匹配

### 问题描述

- 报错 `/lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`

### 原因

- 系统自带 libstdc++ 版本过低，无法满足新编译的 CUDA 扩展需求。

### 解决办法

```sh
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

验证：

```sh
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX
```
应包含 `GLIBCXX_3.4.29`。

---

## 3. Python 运行时类型错误

### 问题描述

- `TypeError: unsupported operand type(s) for divmod(): 'Tensor' and 'int'`

### 解决办法

找到相关代码（如 `train.py`）：

```python
t_m, t_s = divmod(float(remain_time), 60)
# 或
t_m, t_s = divmod(remain_time.item(), 60)
```

---

## 4. 相对位置索引断言失败

### 问题描述

- `AssertionError: assert (relative_position_index >= 0).all()`
- 打印发现 `relative_position_index` 出现 -1。

### 原因

- 浮点误差或边界点导致索引为负。

### 解决办法

在 `/home/hy/projects/Stratified_Transformer/model/stratified_transformer.py` 的 `WindowAttention.forward` 方法中：

```python
relative_position_index = (relative_position + 2 * self.window_size - 0.0001) // self.quant_size
relative_position_index = relative_position_index.clamp(min=0)
```

---

## 5. 训练时大量 WARNING: batch_size shortened from 8 to 1...

### 问题描述

- 数据集单个样本点数很大，超过 `max_batch_points` 限制，collate_fn 自动缩小了 batch_size。

### 解决办法

- 这是资源保护提示，不影响训练。
- 如需减少 warning，可适当减小 `batch_size` 或 `max_batch_points`。

---

## 6. 总结与命令汇总

### 6.1 卸载所有相关包

```sh
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric torch-points-kernels -y
```

### 6.2 安装兼容版本

```sh
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-geometric==1.7.2
pip install torch-points-kernels==0.6.9 --no-binary :all:
```

### 6.3 升级 libstdc++

```sh
conda install -c conda-forge libstdcxx-ng
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### 6.4 代码修正示例

- Tensor 转 float：
  ```python
  t_m, t_s = divmod(float(remain_time), 60)
  ```
- 相对位置索引 clamp：
  ```python
  relative_position_index = relative_position_index.clamp(min=0)
  ```

---

## 7. 经验教训

- PyTorch 及其 CUDA 扩展必须严格版本对应，优先用官方 wheel 或源码编译。
- 遇到符号找不到、GLIBCXX 报错，优先用 conda 升级依赖库。
- 训练中遇到断言、类型等问题，结合调试输出和 clamp 等手段兜底。
- 记录每一步操作和修改，便于后续环境迁移和复现。

**finished 25/08/09 21:02**

---

## 8.若训练被中断，如何从断点处恢复
用model_last.pth  
```bash
python3 train.py --config config/s3dis/s3dis_stratified_transformer.yaml --resume runs/s3dis_stratified_transformer/model/model_last.pth
```
- config :指定你的配置文件。
- resume :指定上次保存的断点权重文件路径（通常为 model_last.pth）。
这样训练会自动从上次中断的 epoch 继续，日志和权重也会继续追加和保存。


**model_last.pth**  
内容：保存了训练中“最后一次保存时”的模型参数、优化器、调度器等全部断点信息。  
作用：用于断点续训（resume），可以无缝从中断的 epoch 继续训练。  
**model_best.pth**  
内容：保存了训练过程中验证集上**当前最优（mIoU 最大）**的模型参数  。
作用：用于最终推理/测试/部署，一般用于模型评估和实际应用，不建议直接 resume 训练（因为它不一定包含优化器等断点信息）。  

---

## 运行训练脚本时遇到如下问题：
```bash
(stratified_transformer_02) hy@hy:~/projects/Stratified_Transformer$ python3 train.py --config config/s3dis/s3dis_stratified_transformer.yaml
Traceback (most recent call last):
File "train.py", line 30, in <module>
import torch_points_kernels as tp
File "/home/hy/miniconda3/envs/stratified_transformer_02/lib/python3.7/site-packages/torch_points_kernels/init.py", line 1, in <module>
from .torchpoints import *
File "/home/hy/miniconda3/envs/stratified_transformer_02/lib/python3.7/site-packages/torch_points_kernels/torchpoints.py", line 7, in <module>
import torch_points_kernels.points_cpu as tpcpu
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/hy/miniconda3/envs/stratified_transformer_02/lib/python3.7/site-packages/torch_points_kernels/points_cpu.so)
```
解决方法
1. 用conda升级libstdc++
在你的conda环境下执行：
```bash
conda install -c conda-forge libstdcxx-ng
```
2. 确保优先使用conda的**libstdc++**
在终端执行：
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
然后再运行你的训练命令。  
3. 确认**GLIBCXX版本**
执行：
```bash
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX
```
确保输出中有**GLIBCXX_3.4.29**。
