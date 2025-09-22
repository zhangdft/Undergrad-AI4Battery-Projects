#  AI-Native Battery Research Agent — 16周零基础学习路径

> **从 Python 零基础 → 独立构建“能听懂指令、调用工具、生成报告”的电池研究智能体**  
> 适用：无 LLM / Agent / 电池背景的本科生  
> 目标成果：构建一个支持自然语言交互、可调用模拟/预测/数据库工具、自动生成报告的完整 Agent 系统  
> 每周投入：6–8 小时（含视频+编码+作业）

---

##  阶段一：Python + LLM 基础（Week 1–4）

### Week 1：Python 基础 + API 调用
- 安装 Anaconda，熟悉函数、类、模块
- 学习 requests 调用简单 API（如天气、翻译）
- 作业：写一个“电池术语翻译器”，输入英文返回中文解释

### Week 2：本地 LLM 部署（Ollama）
- 安装 Ollama，下载 qwen:1.5b 或 llama3:8b-instruct
- 学习命令行对话、API 调用（http://localhost:11434）
- 作业：用 curl 或 Python 调用模型，回答“什么是SEI膜？”

### Week 3：Prompt 工程基础
- 学习 System Prompt、Few-shot Prompt、Chain-of-Thought
- 设计“电池专家”角色 Prompt
- 作业：让模型用三句话解释“锂枝晶是如何导致短路的”

### Week 4：RAG 与知识库构建
- 安装 ChromaDB，学习文本分块 + 向量化（sentence-transformers）
- 将 5 篇电池 PDF 解析 → 分块 → 存入向量库
- 作业：实现“提问 → 检索 → 生成答案”最小闭环

---

##  阶段二：Agent 框架入门（Week 5–8）

### Week 5：LangChain 工具调用入门
- 安装 langchain，学习 Tool 封装
- 创建“计算器工具”“天气查询工具”
- 作业：封装一个“单位换算工具”（mAh ↔ Ah）

### Week 6：构建第一个电池问答 Agent
- 封装“材料查询工具”（输入化学式，返回 MP 形成能）
- 封装“文献检索工具”（RAG）
- 用 Agent 编排：先查文献，再查数据库，最后总结
- 作业：输入“LiCoO2 的电压是多少？”，返回答案 + 来源

### Week 7：多步推理与记忆
- 学习 ReAct 模式、ConversationBufferMemory
- 实现“追问式对话”：用户问“为什么它稳定？” → Agent 回忆上文 → 调用工具 → 回答
- 作业：实现 3 轮连续对话，Agent 能记住上下文

### Week 8：错误处理与日志记录
- 学习 try-except 封装工具、输出结构化日志（JSON）
- 记录每次工具调用：时间、输入、输出、耗时
- 作业：模拟工具失败，Agent 能优雅降级并提示“数据暂不可用”

---

##  阶段三：联动工具链集成（Week 9–12）

### Week 9：封装模拟调用工具（联动方向1）
- 学习 subprocess 调用 LAMMPS 脚本
- 输入“模拟 Li⁺ 扩散” → 启动脚本 → 读取 MSD 输出 → 返回扩散系数
- 作业：封装成 LangChain Tool，输入自然语言，输出数值

### Week 10：封装材料筛选工具（联动方向2）
- 调用已训练好的 XGBoost 模型或 CGCNN
- 输入“筛选高电压材料” → 返回 Top 3 + cif 路径
- 作业：工具返回结构化 JSON：{name, formula, voltage, stability}

### Week 11：封装健康预测工具（联动方向3）
- 加载 ONNX 模型，输入电池 ID → 返回 SOH + 热风险等级
- 作业：工具支持输入“B0005”，输出 {soh: 0.82, risk: "low"}

### Week 12：构建复合任务 Agent
- 输入：“比较 LiCoO2 和 NMC811 的电压和循环寿命，生成对比报告”
- Agent 自主规划：查电压 → 查寿命预测 → 生成对比表 → 输出 Markdown
- 作业：实现该任务，输出包含表格和结论的报告草稿

---

##  阶段四：报告生成 + 系统整合（Week 13–16）

### Week 13：Jinja2 报告模板引擎
- 学习模板语法：变量、循环、条件
- 设计电池报告模板：含标题、摘要、图表占位、参考文献
- 作业：填充模板，生成 PDF（WeasyPrint）

### Week 14：自动化报告 Agent
- 输入：“生成电池 B0005 的月度健康报告”
- Agent 调用：SOH预测 + 热风险 + 充电建议 → 填充模板 → 输出 PDF
- 作业：输出 `B0005_monthly_report.pdf`

### Week 15：项目整合与 GitHub 提交
- 整理：知识库、工具、Agent、前端界面、报告模板
- 编写 README，说明 `python run_agent.py --query "xxx"` 用法
- 作业：提交仓库，确保克隆后 `bash setup.sh` + `python demo.py` 可跑通

### Week 16：成果展示与 Peer Review
- 制作演示视频（3分钟）：展示自然语言提问 → Agent 执行 → 报告生成全过程
- 小组互评 + 导师点评
- 优秀项目推荐：AI Agent 竞赛 / 智能科研工具创新大赛

---
