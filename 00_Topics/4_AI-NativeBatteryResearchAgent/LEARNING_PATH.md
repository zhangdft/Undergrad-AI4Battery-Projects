#  AI-Native Battery Research Agent — 16周零基础学习路径 

> **从 Python 零基础 → 独立构建“能听懂指令、调用工具、生成报告”的电池研究智能体**  
> 适用：无 LLM / Agent / 电池背景的本科生  
> 目标成果：构建一个支持自然语言交互、可调用模拟/预测/数据库工具、自动生成报告的完整 Agent 系统  
> 每周投入：6–8 小时（含视频+编码+作业）

---

##  阶段一：Python + 文献数据基础（Week 1–3）

### Week 1：Python 基础 + API 调用
- 安装 Anaconda，熟悉函数、类、模块
- 学习 requests / BeautifulSoup 基础（用于爬取公开论文摘要）
- 作业：从 ScienceDirect 或 Springer 搜索 “lithium metal electrolyte”，保存前5篇论文标题+摘要到 CSV

### Week 2：PDF 文本提取 + 结构化处理
- 学习 PyPDF2 / pdfplumber 提取 PDF 正文
- 学习正则表达式（regex）提取关键句（如 “CE = 99.2%”, “cycles: 200”）
- 作业：对一篇锂金属电解液论文PDF，提取：电解液成分、电流密度、库伦效率、循环次数”

### Week 3：构建本地文献数据库
- 学习 SQLite / CSV 存储结构化数据
- 设计表结构：paper_id, title, electrolyte_formula, ce, cycles, current_density, additives, doi
- 作业：手动录入5篇论文数据，编写查询函数

---

##  阶段二：构建文献智能体（Week 4–8）

### Week 4：LLM 本地部署 + Prompt 工程
- 安装 Ollama，运行 Qwen2.5-1.5B 或 Llama3-8B-Instruct
- 学习 Prompt 设计：让 LLM 从段落中提取结构化数据”
- 作业：输入一段论文文字，让 LLM 输出 JSON：{"electrolyte": "...", "ce": 99.1, "cycles": 150}

### Week 5：RAG 系统搭建
- 安装 ChromaDB，将 Week 3 的文献数据向量化
- 实现：用户提问 → 检索最相关论文段落 → LLM 生成答案
- 作业：提问“哪些论文用了LiNO3添加剂？”，返回相关论文标题+摘要片段

### Week 6：工具调用 + Agent 工作流
- 使用 LangChain / LlamaIndex 构建 Tool：搜索论文返回列表 → 调用LLM提取性能数据
- 实现 ReAct 工作流：思考 → 调用工具 → 整合结果
- 作业：输入“找出库伦效率>99.5%的电解液配方”，Agent自动检索+提取+汇总

### Week 7-8：期中项目交付 — LMB文献智能体 MVP
- 作业：用户输入自然语言问题（如“找含FEC添加剂且循环>200次的电解液”）, Agent 自动检索本地文献库 → 调用LLM提取性能 → 返回结构化结果
- 系统架构图 + 准确率评估


---

##  阶段三：知识图谱构建（Week 9–12）

### Week 9：知识图谱基础 + 三元组抽取
- 学习知识图谱概念：实体、关系、属性
- 用 LLM 从论文句子中抽取三元组（如 `<LiTFSI, improves, Coulombic Efficiency>`）
- 作业：从5篇论文中抽取20个三元组，保存为 CSV（head, relation, tail）

### Week 10：图数据库入门
- 使用 `NetworkX` 构建本地图谱（轻量，适合教学）
- 或安装 `Neo4j Desktop`（可视化更强）
- 作业：导入三元组，实现“查找所有与FEC相关的性能指标”

### Week 11：图谱增强查询 + 语义推理
- 实现路径查询：`FEC → improves → CE → enables → Long Cycle Life`
- 用 LLM 生成自然语言解释（如“FEC通过形成稳定SEI提升库伦效率，从而延长循环寿命”）
- 作业：输入“为什么FEC能提升锂金属电池性能？”，返回图谱路径 + 自然语言解释

### Week 12：图谱与RAG融合
- 将知识图谱作为“结构化知识源”接入RAG系统
- 查询时同时检索：文献段落 + 图谱路径 → LLM 综合生成答案
- 作业：对比“纯RAG” vs “RAG+图谱”在复杂问题上的回答质量（如“比较FEC和LiNO3的作用机制”）

---

##  阶段四：知识图谱增强LLM系统整合（Week 13–16）

### Week 13：系统架构整合
- 设计统一接口：
  - 输入：自然语言问题
  - 处理：RAG检索 + 图谱查询 + LLM推理
  - 输出：结构化数据 + 自然语言解释 + 参考文献
- 作业：绘制系统架构图，标注各模块数据流

### Week 14：构建Gradio交互界面
- 安装 Gradio，构建聊天机器人界面
- 支持：
  - 文本输入问题
  - 显示答案 + 图谱路径图（NetworkX绘图）
  - 下载参考文献列表（BibTeX或CSV）
- 作业：实现最小可运行Web界面

### Week 15：评估与优化
- 设计20个测试问题（涵盖简单查询、对比、机制解释）
- 人工评估回答准确性、完整性、可解释性
- 优化Prompt或图谱结构提升效果
- 作业：提交评估报告 + 优化前后对比

### Week 16：期末项目交付 + 展示
-  **交付物**：
  - GitHub 仓库（含代码、数据、模型配置、README）
  - Gradio 可运行应用（`app.py`）
  - 项目报告 PDF（含架构图、测试结果、反思）
  - 5分钟演示视频（录屏+讲解）

---

## 附：推荐工具与资源

| 类别 | 推荐工具 | 说明 |
|------|----------|------|
| LLM 本地运行 | Ollama + Qwen2.5-1.5B | 中文能力强，CPU可跑 |
| Agent框架 | LlamaIndex > LangChain | 更轻量，更适合RAG+工具 |
| 向量数据库 | ChromaDB | 本地运行，API简单 |
| 图谱工具 | NetworkX（教学） / Neo4j Desktop（展示） | 后者有可视化界面 |
| PDF解析 | pdfplumber > PyPDF2 | 保留文本位置，适合表格提取 |
| 学术API | Materials Project, PubChem | 获取材料基础属性 |
| 前端 | Gradio | 快速构建AI对话界面 |

---