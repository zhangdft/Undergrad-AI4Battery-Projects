#  AI-Accelerated Materials Discovery — 16周零基础学习路径

> **从 Python 零基础 → 独立完成“AI筛选/生成电池材料”全流程项目**  
> 适用：无 ML / 材料 / 电化学背景的本科生  
> 目标成果：构建一个能从数据库筛选 Top 材料 + 生成新结构 + 输出推荐报告的完整系统  
> 每周投入：6–8 小时（含视频+编码+作业）

---

##  阶段一：Python + 材料数据基础（Week 1–4）

### Week 1：Python 基础 + Jupyter
- 安装 Anaconda，熟悉 Jupyter Notebook
- 学习变量、循环、函数、列表、字典
- 作业：写一个“元素周期表查询器”，输入原子序数返回元素名和族

### Week 2：Pandas + 数据清洗
- 学习 DataFrame、筛选、排序、缺失值处理
- 加载 CSV 格式电池数据（如电压、容量）
- 作业：清洗 NASA 电池数据，提取循环次数 > 500 的样本

### Week 3：材料数据库初探（Materials Project）
- 注册 MP API Key，学习 pymatgen 基础
- 查询 LiCoO2 的晶格参数、能带、形成能
- 作业：下载 5 个正极材料结构，保存为 cif 文件

### Week 4：材料特征工程入门
- 学习 matminer 提取材料特征（如原子半径均值、电负性方差）
- 用 pandas 构造特征表
- 作业：为 10 个材料构造“平均原子质量”“最大配位数”等 5 个特征

---

##  阶段二：机器学习建模（Week 5–8）

### Week 5：Scikit-learn 回归入门
- 学习线性回归、决策树、随机森林
- 预测“形成能” vs. “原子数密度”
- 作业：用 RandomForest 预测材料带隙，输出 R² 分数

### Week 6：模型评估与调参
- 学习 train_test_split, cross_val_score, GridSearchCV
- 调整 n_estimators, max_depth
- 作业：对比 3 种模型在“电压预测”任务上的 MAE

### Week 7：材料专用图神经网络（CGCNN）
- 安装 cgcnn / megnet
- 加载预训练模型，预测新结构的形成能
- 作业：输入 LiFePO4.cif，输出预测形成能，对比 MP 真实值

### Week 8：构建第一个筛选器
- 组合：MP API → pymatgen → matminer → RandomForest
- 输入条件，输出排序候选列表
- 作业：筛选“电压 > 4.0V 且含 Co”的材料，输出 Top 5

---

##  阶段三：生成模型 + 优化（Week 9–12）

### Week 9：生成模型概念（VAE/GAN）
- 学习变分自编码器基础（不推公式，重应用）
- 用 pymatgen 将晶体编码为向量
- 作业：将 5 个 cif 文件编码为 16 维向量

### Week 10：训练条件 VAE（cVAE）
- 使用简单数据集（如 MNIST 或 2D 晶格点）
- 输入“条件标签”，输出生成样本
- 作业：训练 cVAE，输入“含 Li=1”，生成含锂结构编码

### Week 11：结构解码与有效性检查
- 将生成向量解码为晶体结构（需满足化学价、空间群）
- 用 pymatgen 检查结构是否合理
- 作业：生成 3 个结构，检查并保存有效的 cif 文件

### Week 12：贝叶斯优化入门（Scikit-Optimize）
- 学习高斯过程、期望提升（EI）
- 优化“虚拟材料”的“电压 + 稳定性”组合目标
- 作业：在 2D 参数空间中，用 10 步找到最优解

---

##  阶段四：项目整合 + 成果输出（Week 13–16）

### Week 13：构建“筛选-生成-验证”闭环
- 筛选 Top 10 → 生成相似结构 → 用 CGCNN 预测性能 → 反馈
- 作业：实现一个最小闭环，输出增强后候选列表

### Week 14：自动化报告生成
- 用 Jinja2 + WeasyPrint 生成 PDF 报告
- 包含：筛选条件、Top 5 材料、性能对比图、生成结构预览
- 作业：输出 `material_report.pdf`

### Week 15：项目整合与 GitHub 提交
- 整理代码、数据、结果、报告
- 编写完整 README，说明运行方法
- 作业：提交 GitHub 仓库，确保 `python run_all.py` 可复现结果

### Week 16：成果展示与 Peer Review
- 制作 1 页海报或 3 页 PPT
- 小组互评 + 导师点评
- 优秀项目推荐竞赛/毕设

---