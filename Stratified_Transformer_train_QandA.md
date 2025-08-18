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

## 6. 总结与命令汇总（暂时不要参考，仅供个人记录）

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
<mark>可以不修改yaml文件，直接用命令行制定断点文件 / 或者不执行本条命令（指定断点文件的命令）而将yaml文件中train中resume更改为model_last权重文件，保存修改后执行首次启动时使用的启动训练命令即可</mark>


**model_last.pth**  
内容：保存了训练中“最后一次保存时”的模型参数、优化器、调度器等全部断点信息。  
作用：用于断点续训（resume），可以无缝从中断的 epoch 继续训练。  
**model_best.pth**  
内容：保存了训练过程中验证集上**当前最优（mIoU 最大）**的模型参数  。
作用：用于最终推理/测试/部署，一般用于模型评估和实际应用，不建议直接 resume 训练（因为它不一定包含优化器等断点信息）。  

---

## 9.运行训练脚本时遇到如下问题：
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

---

## 10.关于训练经常中断导致执行时间太长、效率低下的发现
1.第一条  
<img width="1460" height="796" alt="image" src="https://github.com/user-attachments/assets/48061069-4e91-400a-ba0b-430bb2d86a67" />  
于2025年8月12日下午13：35发现，训练进度是epoch80/100，资源监控网页显示GPU从未停止使用  
根据上述日志的时间记载  
发现训练暂停于**格林尼治时间3：44（北京时间11：44）**（仍然在训练epoch80/100）  
继续进行于**格林尼治时间5：33（北京时间13：33）**（从11：44暂停的位置继续训练）
继续训练时未通过命令进行训练的启动，而是直接输入ssh密码

2.第二条  
<img width="1516" height="378" alt="image" src="https://github.com/user-attachments/assets/83851613-613f-4a5e-b19c-e843ac7de235" />
2025年8月12日早8：00发现训练又被暂停  ，GPU早已结束使用
根据日志中显示的格林尼治时间推断：  
训练终止于**格里尼治时间2025年8月11日 13：31（北京时间21：31）**
训练继续于**格里尼治时间2025年8月12日 00：07（北京时间08：07）**
继续训练的时候先输入了ssh密码，再通过bash命令进行了训练的启动

3.第三条
<img width="1497" height="457" alt="image" src="https://github.com/user-attachments/assets/65b0c8d9-1194-43ae-80a0-8994eb125d6e" />
2025年8月11日早发现训练暂停，GPU早已结束使用
根据日志中显示的格林尼治时间推断：
训练终止于**格林尼治时间2025年8月10日 13：48（北京时间21：48）**
训练继续于**格林尼治时间2025年8月11日 06：15（北京时间14：15）**
继续训练的时候先输入了ssh密码，再通过bash命令进行了训练的启动

---

## 11.关于mIoU=7%的排错
找到问题的原因：
- 问题出在读取预处理后的数据时，读取的是第7列，而实际预处理后的数据（官方格式）中标签存放在第9列，修改前代码如下：
<img width="739" height="624" alt="image" src="https://github.com/user-attachments/assets/dd3bab23-e34a-4b16-a1e5-0b94c5865d8f" />

- 修改后代码如下：
<img width="753" height="620" alt="image" src="https://github.com/user-attachments/assets/0da82798-25f1-4275-a9db-4844e1ccb96d" />

- **data_prepare_output**中数据内容：
  
  ```text
    列号	含义	说明
     0	   x	  点的x坐标
     1	   y	  点的y坐标
     2	   z	  点的z坐标
     3	   r	  红色通道（0-255）
     4	   g	  绿色通道（0-255）
     5	   b	  蓝色通道（0-255）
     6	  label	语义类别标签（整数）
  ```
  
- **data_prepare_output_converted**中数据内容：

  ```text
  列号	含义	说明
   0	   x	点的 x 坐标
   1	   y	点的 y 坐标
   2	   z	点的 z 坐标
   3	   r	红色通道（0-255）
   4	   g	绿色通道（0-255）
   5	   b	蓝色通道（0-255）
   6	   ?	可能为 instance id 或保留位 / 插值，为了转换为9列数据
   7	   ?	可能为 instance id 或保留位 / 插值，为了转换为9列数据
   8	 label	语义类别标签（整数）
  ```
- 使用**check_labels.py**排查data_prepare_output_converted文件见下是否第9列全部为labels，如果都是则返回True，如果有的不是，则返回该文件名

---

## lib文件夹下代码缺失
由于上传限制文件大小，部分文件未上传，可去原作者仓库中克隆
