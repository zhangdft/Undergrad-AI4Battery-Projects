# 本科生机器学习电池模拟训练项目

> 从入门到实践 · 三阶段科研能力培养体系  
> 融合 **电化学分析 + 多尺度计算模拟 + 机器学习方法**

本课程项目分为三类，循序渐进，帮助学生构建完整的“电池材料智能设计”能力：

| 类别 | 目标 | 适合人群 | 项目示例 |
|------|------|----------|----------|
| **方向介绍**<br>(Topics) | 四个主要方向介绍 | 查看感兴趣的方向及培养方案 |
| **入门项目**<br>(Getting Started) | 掌握基础工具与方法，建立科研规范 | 大一新生 | Python 数据分析、MD/DFT基础计算模拟等 | [`LEARNING_PATH.md`](./00_Topics/1_AI-DrivenMultiscaleSimulation/LEARNING_PATH.md)
| **进阶项目**<br>(Advanced) | 深化理解，自主设计计算方案 | 有较强理论基础 | DFT界面计算、ML势函数训练等 |
| **实践项目**<br>(Practical) | 解决真实科研/工程问题，输出可发表成果 | 有较强理论基础和实践经验 | SOX预测、电池失效机理解析、固态电池设计 | 

---

##  如何开始？

1. 阅读[`Topics`](./00_Topics) 了解相应方向和要求
2. 参考方向相应的README.md和LEARNING_PATH.md，选择适合你的项目路径
3. 从入门项目开始，逐步挑战更高难度
4. 所有项目均提供完整说明文档、依赖列表、模板代码

---

##  项目要求

1. 每周至少投入6小时，暑假投入2周以上，寒假投入1周以上
2. 入门项目采用作业制，按时提交入门训练项目的进展，才可开展后续训练
3. 作业提交至[`Submissions`](./Submissions)
4. 所有作业均提供按照项目要求提交
5. 项目可以随时自愿退出，另外，半年内无法完成入门项目，也视为退出

---

## 交流共享 · 技术沙龙与科研社区

我们每月组织 **本科生 × 研究生技术沙龙**，聚焦：

- 电池模拟技巧 / ML 最新论文 / 项目卡点互助 / 工具链分享
- 定期邀请 **学界/业界大咖** 做专题报告
- 线上社区永久开放：微信群 + GitHub Discussions 

---

##  项目目录结构

- ` 00_Topics/` —— 细分方向，了解内容和培养路线 (持续建设中)
- ` 01_GettingStarted/` —— 手把手入门，重在规范与基础 (陆续更新中)
- ` 02_Advanced/` —— 强化科研思维，鼓励自主探索
- ` 03_Practical/` —— 面向真实问题，产出科研级成果
- ` Common_Resources/` —— 图表模板、文献引用、学习链接
- ` Submissions/` ——  训练项目作业提交

---

##  工具与规范

- **代码规范**：PEP8 + Google Docstring + Black/Flake8
- **图表标准**：Nature 风格，≥300 dpi，支持矢量图
- **文档要求**：每个项目含 `README.md` + 技术报告 + PPT
- **版本控制**：Git + GitHub，鼓励使用 Branch + Pull Request

---

##  第一次作业：兴趣方向与时间规划

>  **如果你是通过 GitHub Classroom 加入本课程的学生，请按以下步骤提交 Assignment 1。**

### 📌 作业要求
请在 `Submissions/` 目录下创建**以你用户名命名的文件夹**，并在其中提交一个 `README.md`，内容包括：

1. **个人兴趣方向**
   - 个人简介（姓名、年级、学院等）
   - 从上述课题中选择 1 个你最感兴趣的方向
   - 简要说明原因（如：有 Python 基础 / 对材料科学感兴趣等）

3. **时间规划**  
   - 本学期每周可投入小时数（如：6–8 小时）
   - 关键里程碑计划（如：第 4 周完成python基础训练等）

4. **（可选）技能与背景**  
   - 编程语言、相关课程、项目经验等

### 提交路径示例

Submissions/
└── your-username/      ← 例如：zhangsan
└── README.md              ← 你的作业内容写在这里

###  提交步骤
1. 克隆你的个人作业仓库（由 GitHub Classroom 自动创建）
2. 创建目录并编写 `README.md`：
   ```bash
   mkdir -p Submissions/your-github-username
   # 编辑 Submissions/your-github-username/README.md
提交代码：
git add .
git commit -m "Submit Assignment 1: Interest & Plan"
git push origin main
提示：你的 GitHub 用户名可在 github.com 右上角头像处查看。

---

## 联系方式

- 指导教师：张宝 特聘研究员
- 联系方式：zhangbao@uestc.edu.cn
- 办公时间：每周日 10:00–12:00 @创新中心C209
