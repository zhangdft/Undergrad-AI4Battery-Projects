##  学习路径：零基础 → 独立完成 ML 势函数 + MD 模拟（16周）

> **适用对象**：无 Python / ML / 计算材料 / 电化学 基础的本科生  
> **目标成果**：能独立训练一个简单 ML 势函数（如 LJ 或水分子体系），在 LAMMPS 中运行模拟，分析轨迹并输出扩散系数/能量图  
> **每周建议投入**：6–8 小时（含视频+编码+作业）

---

###  阶段一：Python 与科学计算筑基（Week 1–4）

#### Week 1：Python 基础语法 + Conda 环境搭建 + Jupyter Notebook
- 安装 Miniconda + VS Code / Jupyter Lab / Qoder，搭建相关环境
- 学习变量、循环、函数、列表、字典
- 作业：写一个“电池材料元素筛选器”（输入原子序数，输出是否为锂/钠/钾）

#### Week 2：NumPy + Matplotlib 数据处理与绘图
- 学习数组操作、切片、广播
- 绘制折线图、散点图、直方图
- 作业：读取 CSV 格式电池电压数据，绘制充放电曲线

#### Week 3：Pandas + 文件读写
- 学习 DataFrame、数据筛选、groupby
- 读写 txt/csv/json 文件
- 作业：整理 NASA 电池数据集，提取某电池的循环寿命数据并绘图

#### Week 4：基础电化学 + 计算材料概念
- 学习：锂离子电池工作原理、SEI、离子电导率、扩散系数
- 了解：DFT、MD、势函数、力场基本概念（不深究数学）
- 作业：读书笔记

---

###  阶段二：机器学习入门 + 材料建模操作（Week 5–8）

#### Week 5：机器学习基础（监督学习）
- 学习：训练集/测试集、损失函数、过拟合
- 安装matminer， 学习其基本用法
- 作业：用ML算法预测材料“体积模量”，输出特征重要性排序图

#### Week 6：神经网络入门（PyTorch）
- 了解多层感知机、前向/反向传播、激活函数、权重衰退、暂退法等基本概念
- 了解深度学习网络结构
- 使用 MoleculeNet 中的 ESOL 数据集，构建简单 MLP，进行分子溶解度预测

#### Week 7：材料原子级建模基础
- 安装Materials Studio，学习相关可视化建模 
- 安装 ASE/Pymatgen，学习构建晶胞、分子、表面
- 导出 xyz / cif 文件
- 作业：构建 Li|Li2O的任意一个界面结构，可视化并保存为图片

#### Week 8：ASE + 材料计算 初体验
- 学习 ASE 调用 相关计算模块计算能量/力
- 作业A：ASE + Genetic Algorithm 优化 Cu-Au 合金 convex hull
- 作业B：pymatgen + VASP 计算单晶 Si 的能带结构

---

###  阶段三：DeePMD机器学习势函数入门（Week 9–12）

#### Week 9：DeePMD-kit 安装与环境配置
- 安装 deepmd-kit 
- 配置 PATH 和 Python 接口
- 作业：验证：dp -h, 确认安装成功

#### Week 10：理解 ML 势函数输入数据格式
- 学习 `type.raw`, `coord.raw`, `energy.raw`, `force.raw`
- 用 ASE 生成 50 个构型的 “Li2O” 体系（扰动位置 + 体积缩放）
- 作业：手动生成 5 个构型，包含能量和力（可用 EMT 势函数快速估算）,用 dp train --init-model 初始化模型

#### Week 11：训练第一个 ML 势函数 — Li金属体系
- 下载相关数据集
- 编写 `input.json`，设置网络结构、训练参数
- 运行 `dp train`，监控 loss 曲线
- 作业：调整神经网络宽度/深度，观察训练收敛速度

#### Week 12：用训练好的模型在 LAMMPS 中推理
- 编译 LAMMPS + DeePMD 插件
- 在 LAMMPS input 文件中调用 `pair_style deepmd`
- 作业：运行 NVT 模拟，输出轨迹文件（.lammpstrj）

---

###  阶段四：分析 + 输出 + 项目整合（Week 13–16）

#### Week 13：轨迹分析基础（MSD / RDF）
- 用 OVITO 或 MDAnalysis 计算均方位移（MSD）
- 用 MSD 计算扩散系数 D
- 作业：绘制 MSD-t 曲线，拟合得到 D 值

#### Week 14：可视化与报告生成
- 用 Matplotlib 绘制：能量、温度、扩散系数图
- 用 Markdown + Jinja2 生成简易报告模板
- 作业：输出 PDF 格式“我的第一个 ML 势函数模拟报告”

#### Week 15：项目整合 — 从零复现全流程
- 自选简单体系（如：Si, H2O, NaCl）
- 独立完成：数据生成 → 训练 → 模拟 → 分析 → 报告
- 提交 GitHub 仓库（含 README + 代码 + 图片 + 报告）

#### Week 16：成果展示 + Peer Review
- 制作 3 页 PPT / 1 页海报，展示项目

---

##  交付成果（期末）

学生需提交一个 GitHub 仓库，包含：

- `README.md`：项目说明、方法、结果图
- `data/`：训练数据（至少50个构型）
- `scripts/`：数据生成、训练、模拟、分析脚本
- `results/`：loss曲线、能量图、MSD图、扩散系数
- `final_report.pdf`：整合所有结果的简易科研报告

---


##  温馨提示

- 不要求理解 DeePMD 的数学推导，重点在“会用、会调、会分析”
- 允许“模仿-修改-创新”路径，先跑通示例，再改参数，最后自定义体系
- 支持组队（≤2人），合理分工

---