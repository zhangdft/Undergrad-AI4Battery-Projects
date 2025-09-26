#  Data-Driven Battery Intelligence — 16周零基础学习路径

> **从 Python 零基础 → 独立完成“SOH预测 + 智能充电”全流程项目**  
> 适用：无 ML / 电化学 / 信号处理背景的本科生  
> 目标成果：构建一个能预测电池寿命、优化充电策略的完整系统  
> 每周投入：6–8 小时（含视频+编码+作业）

---

##  阶段一：Python + 电池数据入门（Week 1–4）

### Week 1：Python 基础 + Jupyter
- 安装 Miniconda，熟悉 Jupyter Notebook
- 学习变量、循环、函数、列表、字典
- 作业：写一个“电池编号查询器”，输入 ID 返回型号和标称容量

### Week 2：Pandas 数据加载与清洗
- 学习读取 CSV/HDF5，处理缺失值、异常值
- 加载 NASA 电池数据 / 或者自测电池数据
- 作业：提取某一个电池的第 1~100 次循环的充放电数据，绘制容量-电压、容量-循环等曲线

### Week 3：电池基础概念 + 特征观察
- 学习：SOH/SOC 定义、容量衰减、内阻、跳水
- 观察不同循环次数下电压曲线的“平台偏移”“斜率变化”
- 作业：统计不同电池容量跳水位置

### Week 4：时序特征工程入门
- 学习滑动窗口、差分、移动平均
- 计算 dQ/dV, dV/dt, 平台电压均值
- 作业：进行ICA分析，提取相关特征，绘图观察趋势

---

##  阶段二：ML 与 DL 初探 — 极早期寿命预测（Week 5–8）

### Week 5：传统 ML 回归模型
- 学习线性回归、随机森林、XGBoost
- 构建特征：前 N 个循环的容量衰减速率、内阻增长、充电时间变化等
- 作业：用前 100 个循环预测 EOL

### Week 6：深度学习时序模型
- 学习 RNN/LSTM 基础概念
- 用 PyTorch 构建简单 LSTM
- 作业：训练 LSTM，对比其与 XGBoost等ML方法，在相同数据上的表现

### Week 7：Transformer与自动特征
- 使用 tsfresh 自动生成 50+ 时序特征
- 尝试 Time Series Transformer
- 作业：构建三类模型（XGBoost + LSTM + Transformer），在 NASA 数据上对比 EOL 预测误差

### Week 8：构建第一个预测系统
- 整合：数据加载 → 特征提取 → 模型预测 → 结果绘图
- 作业：极早期寿命预测

---

##  阶段三：强化学习 + 电池仿真（Week 9–12）

### Week 9：RL 基础概念 + Gym 环境初体验
- 强化学习基本框架：Agent、Environment、State、Action、Reward、Episode
- 安装 gym（或 gymnasium），了解 env.reset() / env.step() / env.render()
- 学习离散动作空间 vs. 连续动作空间
- 读书笔记

### Week 10：训练 PPO Agent 
- 安装 Stable-Baselines3
- 理解 PPO（Proximal Policy Optimization）
- 作业：用 SB3 的 PPO 算法训练 CartPole-v1 智能体

### Week 11：PyBaMM 电池仿真入门
- 安装 PyBaMM，运行一个标准 CC-CV 充电仿真
- 修改充电电流（如 0.5C vs. 2C），观察仿真输出的容量衰减
- 作业：生成 3 种不同充电策略下的 50 次循环仿真数据

### Week 12：用真实模型预测仿真电池寿命
- 将之前训练好的 XGBoost/LSTM 模型，应用于 PyBaMM 生成的仿真数据
- 作业：验证模型在仿真数据上的泛化能力

---

##  阶段四：虚拟电池智能充电系统（Week 13–16）

### Week 13：PyBaMM 仿真 + 极早期寿命预测
- 学会设置不同老化模型
- 学会提取仿真输出：容量、电压、内阻、析锂风险等
- 加载预训练的模型，对仿真生成的前 10 个循环数据提取特征，调用模型预测“总寿命”（EOL）
- 作业：对 3 种策略（0.5C / 1.0C / 2.0C）分别预测 EOL，并评估充放电效率

### Week 14：轻量级 RL 策略优化
- 以EOL、充放电效率构建奖励，轻量级 RL，优化充电策略

### Week 15~16：成果展示与 Peer Review
- 整理代码、数据、结果、报告
- 编写完整 README，说明运行方法
- 作业：提交 GitHub 仓库（含 README + 代码 + 图片 + 报告）
- 制作 1 页系统架构图 + 3 页成果 PPT


---

##  交付成果（期末）

学生需提交一个 GitHub 仓库，包含：

- `README.md`：项目说明、方法、结果图
- `data/`：训练数据（至少50个构型）
- `scripts/`：数据生成、训练、模拟、分析脚本
- `results/`：所有结果分析
- `final_report.pdf`：整合所有结果的简易科研报告

---


##  温馨提示

- 不要求理解 DL/RL/PyBaMM 的数学推导，重点在“会用、会调、会分析”
- 允许“模仿-修改-创新”路径，先跑通示例，再改参数，最后自定义体系
- 支持组队（≤2人），合理分工

---
