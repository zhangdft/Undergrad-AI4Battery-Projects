# 电池测试原始数据处理与可视化分析

## 项目目标
使用 Python 对 Neware 电池测试设备导出的 `FullCell.nda` 原始数据进行处理与分析，绘制符合 Nature 及其子刊出版标准的科研图表，包括：
- 充放电曲线（Voltage vs. Capacity）
- 容量衰减曲线（Cycle Number vs. Capacity Retention）
- 微分容量分析曲线（dQ/dV vs. Voltage）
- 图表风格统一、配色专业、标注完整、分辨率 ≥ 300 dpi，符合 Nature 及其子刊系列的发表标准

## 技术要求
- 使用 `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `NewareNDA`, `neware_reader` 等库。
- 实现自动化数据读取、清洗、分段（充/放电）、平滑处理（Savitzky-Golay）、微分计算。
- 输出矢量图（PDF/SVG）和高分辨率位图（PNG/TIFF）。
- 代码结构清晰，函数模块化，含完整 docstring 和 README.md 说明文档。

## 推荐参考资料
- **查全性，《电极过程动力学导论》（第三版），科学出版社，2002**  
  → **中国电化学领域“圣经级”教材**，系统讲解电极/溶液界面、双电层、吸附、极化、动力学控制步骤等核心概念，是理解电池充放电过程、dQ/dV 分析、界面反应机制的理论基石。  
  → 重点阅读：第1–4章、第6章。
  
- **查全性，《化学电源选论》，武汉大学出版社，2005**  
  → 从电化学热力学与动力学角度，系统分析各类化学电源（含锂离子电池）的工作原理、能量密度限制、失效机制与发展方向。  

- 锂离子电池的ICA和DVA分析

## 时间安排与交付物

### 时间节点
- **启动时间**：即日起
- **中期检查**：10月中旬（提交初步代码+展示结果）

### 最终提交（10月中旬）
- 完整项目代码（含说明文档和依赖列表）
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
