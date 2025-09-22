#  AI-Accelerated Materials Discovery
## 人工智能加速的电池材料高通量筛选与生成设计

> **用机器学习 + 材料数据库 + 生成模型，快速发现下一代高性能电极/电解质材料**

---

##  核心目标

- 掌握高通量材料筛选流程：从数据库检索 → 特征工程 → 性能预测 → 排序推荐
- 学会构建材料生成模型（如 CGCNN、MEGNet、GAN），设计“未被发现”的新材料
- 实现“AI 驱动闭环”：预测 → 生成 → 筛选 → 反馈优化，逼近目标性能（如电压 > 4.5V, 稳定性 > 0.5eV）
- 输出可复用代码库 + 材料推荐报告，为实验合成提供优先级清单

---

##  知识与技术储备

###  基础理论（建议预习）

- 材料科学基础：晶体结构、能带、形成能、电压平台、离子迁移能垒
- 机器学习基础：回归/分类、特征工程、交叉验证、超参数调优
- 电化学基础：电极电位、离子电导率、界面稳定性、循环寿命关联指标

### ️ 技术栈

- **材料数据库**：Materials Project API / OQMD / Battery Archive
- **特征提取**：pymatgen / matminer
- **ML 模型**：scikit-learn / XGBoost / CGCNN / MEGNet
- **生成模型**：GAN / VAE / Diffusion（可选）
- **优化引擎**：Scikit-Optimize / Ax / Optuna
- **可视化**：Plotly / Seaborn / pymatgen.structure_view

---

## ‍ 适合学生

- 对新材料发现、AI for Science、数据驱动科研感兴趣
- 有 Python 基础（会写函数、读文件、画图），无 ML/材料背景也可
- 适合大二~大四本科生，可单人或2人组队
- 无需实验设备 —— 全程基于公开数据库 + 计算模拟

---


## 📬 联系我们

- 提交 Issue：[点击提问](https://github.com/zhangdft/Undergrad-AI4Battery-Projects/issues)
- 讨论区：[Discussions](https://github.com/zhangdft/Undergrad-AI4Battery-Projects/discussions)
- 导师邮箱：zhangdft@uestc.edu.cn