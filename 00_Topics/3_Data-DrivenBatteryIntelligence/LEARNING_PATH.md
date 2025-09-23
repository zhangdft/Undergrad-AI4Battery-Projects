#  Data-Driven Battery Intelligence — 16周零基础学习路径

> **从 Python 零基础 → 独立完成“SOH预测 + 智能充电 + 热预警”全流程项目**  
> 适用：无 ML / 电化学 / 信号处理背景的本科生  
> 目标成果：构建一个能预测电池寿命、优化充电策略、预警热失控的完整系统  
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

##  阶段二：机器学习建模（Week 5–8）

### Week 5：Scikit-learn 回归入门
- 学习线性回归、随机森林、XGBoost
- 用循环次数预测 SOH（简单基线）
- 作业：对比 3 种模型在前 80% 数据训练、后 20% 测试的 MAE

### Week 6：LSTM 时序预测入门
- 学习 RNN/LSTM 基础概念
- 用 PyTorch 构建简单 LSTM，输入前 50 个循环 SOH，预测第 51 个
- 作业：调整 hidden_size，观察预测误差变化

### Week 7：高级特征 + Transformer
- 使用 tsfresh 自动生成 50+ 时序特征
- 尝试 Time Series Transformer（HuggingFace 或 PyTorch）
- 作业：对比 LSTM vs. Transformer 在 SOH 预测上的 RMSE

### Week 8：构建第一个预测系统
- 整合：数据加载 → 特征提取 → 模型预测 → 结果绘图
- 作业：输入电池 ID，输出未来 10 个循环的 SOH 预测曲线

---

##  阶段三：强化学习 + 安全预警（Week 9–12）

### Week 9：Gym 环境入门
- 安装 gym，学习 State/Action/Reward 设计
- 构建简化版“电池充电环境”（离散动作：CC1C/CC2C/CV）
- 作业：手动写一个“总是用 CC1C”的策略，观察寿命

### Week 10：训练 PPO Agent 优化充电
- 安装 Stable-Baselines3
- 设计奖励函数：R = -ΔSOH + 1/充电时间
- 作业：训练 1000 步，对比 RL 策略 vs. CC-CV 的 200 次循环后容量

### Week 11：异常检测与热预警
- 学习 Isolation Forest、LSTM Autoencoder
- 用温度、dV/dt 构建特征，检测“异常升温段”
- 作业：在 CALCE 数据中标注 3 个热异常段，训练模型检测并输出预警点

### Week 12：实时仪表盘搭建（Streamlit）
- 安装 streamlit，构建 Web 界面
- 上传数据 → 自动预测 SOH → 显示充电建议 → 热风险指示灯
- 作业：实现一个带“上传按钮 + 预测按钮 + 图表显示”的最小仪表盘

---

##  阶段四：项目整合 + 成果输出（Week 13–16）

### Week 13：模型轻量化与导出
- 学习 ONNX 格式，导出训练好的 LSTM/XGBoost 模型
- 用 ONNX Runtime 加载并推理
- 作业：将 SOH 模型导出为 .onnx，编写推理脚本 `predict_soh.py`

### Week 14：自动化报告生成
- 用 Jinja2 + WeasyPrint 生成 PDF 报告
- 包含：电池信息、预测曲线、充电建议、热风险等级
- 作业：输出 `battery_health_report.pdf`

### Week 15：项目整合与 GitHub 提交
- 整理代码、模型、数据、仪表盘、报告
- 编写完整 README，说明 `python run_dashboard.py` 启动方法
- 作业：提交 GitHub 仓库，确保一键复现

### Week 16：成果展示与 Peer Review
- 制作 1 页系统架构图 + 3 页成果 PPT
- 小组互评 + 导师点评
- 优秀项目推荐：全国大学生嵌入式竞赛 / 智能车竞赛 BMS 赛道

---
