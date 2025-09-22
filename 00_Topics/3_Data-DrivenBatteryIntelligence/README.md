#  Data-Driven Battery Intelligence
## 数据驱动的电池状态预测与智能充电优化

> **用机器学习 + 电池大数据 + 强化学习，构建下一代智能电池管理系统（BMS）核心算法**

---

##  核心目标

- 掌握电池数据采集、清洗、特征工程全流程
- 构建高精度 SOH（健康状态）、SOC（充电状态）预测模型
- 开发基于强化学习的“智能充电策略优化器”，延长寿命、抑制析锂
- 实现电池热失控早期预警模型，提升安全性
- 输出可部署的轻量化模型 + 实时预测仪表盘，对接真实 BMS 系统

---

##  知识与技术储备

###  基础理论（建议预习）

- 电池基础：充放电曲线、容量衰减机制、内阻增长、析锂、热失控
- 信号处理：滑动窗口、FFT、小波变换（用于电压/电流噪声处理）
- 机器学习：时序预测（LSTM、GRU、Transformer）、回归、分类、异常检测
- 强化学习：状态-动作-奖励建模、Q-Learning、PPO（用于充电策略优化）

### ️ 技术栈

- **数据源**：NASA Prognostics Dataset / Oxford Battery Degradation / CALCE
- **数据处理**：pandas / numpy / tsfresh / scipy.signal
- **ML 模型**：scikit-learn / XGBoost / LightGBM / PyTorch (LSTM/Transformer)
- **强化学习**：Stable-Baselines3 (PPO/A2C) / Gym-Battery（自定义环境）
- **可视化**：Plotly / Dash / Streamlit（构建实时仪表盘）
- **部署**：ONNX / Flask / Docker（可选）

---

##  适合学生

- 对电池管理、智能算法、时序数据分析感兴趣
- 有 Python 基础（会写函数、读文件、画图），无电化学/ML 背景也可
- 适合大二~大四本科生，可单人或2人组队
- 无需硬件 —— 全程基于公开数据集 + 仿真环境

---

## 📬 联系我们

- 提交 Issue：[点击提问](https://github.com/zhangdft/Undergrad-AI4Battery-Projects/issues)
- 讨论区：[Discussions](https://github.com/zhangdft/Undergrad-AI4Battery-Projects/discussions)
- 导师邮箱：zhangbao@uestc.edu.cn