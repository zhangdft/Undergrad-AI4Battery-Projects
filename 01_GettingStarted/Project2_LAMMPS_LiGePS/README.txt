# LiGePS电解质离子输运机制研究

## 项目目标
使用 LAMMPS + DeePMD-kit（lammps-dpmd）对硫化物固态电解质 Li₁₀GeP₂S₁₂ (LiGePS) 进行分子动力学模拟，分析其离子输运性质：
- 计算离子扩散系数与活化能（通过 Arrhenius 拟合）
- 分析原子态密度（Density of Atomic States, DOS）
- 构建并可视化锂离子密度分布（Li+ probability density）
- 进行集体跃迁事件识别与统计（Collective Hopping Analysis）

## 技术要求
- 使用 LAMMPS + DeePMD-kit 进行高温/多温度点模拟。
- 利用 `ovito`, `pymatgen`, `MDAnalysis`, `VMD` 等工具进行轨迹分析。
- 编写脚本自动计算 MSD（均方位移）、DOAS（原子态密度）、渗流路径、跃迁事件。
- 输出高质量科研图表（Nature 风格）及3D可视化动图（可选）。
- 代码需模块化、带注释，含 `requirements.txt` 和 `README.md`。

## 推荐参考资料

### LAMMPS 与 DeePMD 模拟
- **LAMMPS 官方手册**：[LAMMPS Documentation](https://docs.lammps.org)  
  → 重点阅读：compute msd, compute vacf, fix nvt/npt, dump custom
- **DeePMD-kit 官方文档**：[DeePMD-kit Documentation](https://docs.deepmodeling.com/projects/deepmd/)  
  → 入门教程、LAMMPS 接口、训练与推理流程
- **玻尔社区**：[Bohrium Community](https://www.bohrium.com/)  
  → 内含大量 DeepMD 的计算和分析案例
- **DP-势函数**：[AI Square](https://www.aissquare.com/models/detail?pageType=models&name=LiGePS-SSE-PBEsol-model&id=34)  
  → 提供 LiGePS 模型的详细介绍与下载

### 固态电解质离子输运分析方法
- He X, Zhu Y, Mo Y. Origin of fast ion diffusion in super-ionic conductors[J]. *Nature Communications*, 2017, 8(1): 15893.  
  → 解释超离子导体中快速离子扩散的起源。
- Wang S, Liu Y, Mo Y. Frustration in super‐ionic conductors unraveled by the density of atomistic states[J]. *Angewandte Chemie International Edition*, 2023, 62(15): e202215544.  
  → 通过原子态密度解析超离子导体中的结构无序现象。

## 时间安排与交付物

### 时间节点
- **启动时间**：即日起
### 最终提交（11月15日）
- 完整项目代码（含说明文档 `README.md` 和依赖列表 `requirements.txt`）
- 技术报告文档（PDF格式，含方法、结果、讨论）
- 展示用PPT
- 所有图表源文件（.py 脚本 + .pdf/.svg/.png 输出）

## 编码规范与文档要求
- 所有 Python 脚本必须符合 **PEP8** 规范（使用 `black` + `flake8` 格式化与检查）。
- 每个函数/类必须包含 **Google Style Docstring**。
- 项目根目录必须包含：
  - `README.md`：项目概述、安装依赖、运行命令、结果说明
  - `LICENSE`（建议 MIT）
  - `requirements.txt` 或 `environment.yml`
- 图表输出文件夹结构清晰（如 `/figures/dQdV/`, `/figures/ChargeDischarge/`）

## 理论学习建议（贯穿项目全程）

### 计算模拟基础（DFT/MD）
- 单斌老师课程： [Bilibili Space](https://space.bilibili.com/1111135013)  
  → 包含 DFT 和 MD 的基础讲解，适合初学者入门。
