
# Bayesian-Informed Optimization (BIO) Framework

## 准备工作
### 1.策略生成
在贝叶斯优化 (BO) 开始之前，在终端执行以下的命令：
```
python Generation.py
```
所有的策略将保存在`policies_raw.csv`中，经过打乱的策略保存在`policies_all.csv`中，其中包含了 924 个策略
### 2.冷启动
在终端执行以下命令，将首次的实验所采集的 50 个策略保存在`data/batch/0.csv`文件中，数据和模型的细节分别保存在`data/bounds/0.pkl`和`data/bounds/0_bounds.pkl`文件中，执行以下命令可完成
```
python Closed_Loop_Opt.py --round=0
```

### 3.闭环优化
第 i 批次的实验已经完成后，实验数据保存在`Testdata/round<i>`文件夹中，通过在终端执行以下的命令将实验数据清洗和归一化后，保存在`data/pred/<i>.csv`文件中：
```
python Datacollection.py --round=i
```
随后，执行以下命令生成第 i+1 批次所需要的实验策略，这些策略保存在`data/batch/<i+1>.csv`文件中：
```
python Closed_Loop_Opt.py --round=i
```
### 4.其他信息
在这项工作中，由于所有的电池测试是在 NEWARE 平台上运行的，使用的工步文件是 xml 格式的文件，当需要在平台上输入大量参数（一轮中需要输入 50×6 个参数）时，我们在`Tools/`中编写了`XML.py`方便快速根据策略的参数生成工步文件

为了得到模拟数据集进行超参数的优化，我们编写了可执行的脚本`simulation.bat`和相关的执行文件`Physical_Prediction.py`和`simulation_policies.py`，根据策略的参数来模拟实验结果的值。这一模型与闭环优化无直接联系，只是生成训练的模拟数据集

我们还编写了关于聚类方法的`Cluster.py`文件，尝试通过简单的机器学习算法来解释闭环优化的结果并编写了一系列文件方便闭环优化数据的可视化

正文中Fig 3的源数据都储存在`Figure_file/`文件夹中

部分代码参考了Attia团队的工作，详见https://github.com/chueh-ermon/battery-fast-charging-optimization