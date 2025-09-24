#  AI-Native Battery Research Agent
## 电池研究智能体

> **用 LLM + Agent 框架 + 工具链，构建一个面向电池领域的 AI 科研智能体系统**

---

##  核心目标

- 掌握 LLM（大语言模型）本地部署与调用（Ollama / Llama3 / Qwen）
- 构建多工具调用 Agent（LangChain / AutoGen），支持：文献检索、数据查询、模拟调用、报告生成
- 实现“自然语言指令 → 自主规划 → 工具执行 → 结果整合 → 报告输出”全流程
- 输出可对话、可扩展、可复现的科研 Agent 系统

---

##  知识与技术储备

###  基础理论（建议预习）

- LLM 基础：Prompt 工程、RAG、Function Calling、Agent 工作流
- 电池知识：关键术语（如 SEI、枝晶、离子电导率）、常见问题（如“哪种电解质最稳定？”）
- 多模态数据对齐/科研自动化思维
- 工具集成：API 封装、参数传递、错误处理、结果解析

### ️ 技术栈

- **LLM 引擎**：Ollama (Llama3/Qwen) 
- **Agent 框架**：LangChain / AutoGen
- **知识库**：ChromaDB + RAG（本地电池文献/FAQ/数据库）
- **工具链集成**：
  - `pymatgen`：查询材料结构
  - `ASE + LAMMPS`：调用模拟（联动方向1）
  - `Materials Project API`：获取材料性能（联动方向2）
  - `NASA 数据接口`：获取循环数据（联动方向3）
  - `Jinja2`：自动生成报告
- **前端交互**：Gradio / Streamlit

---

## ‍ 适合学生

- 对 AI Agent、自动化科研、LLM 应用感兴趣
- 有计算机科学基础，
- 无 LLM/Agent 背景者，需要极强的自驱力
- 可单人或2人组队
- 每周需投入6小时以上，寒暑假需投入一个月

---


##  联系我们

- 提交 Issue：[点击提问](https://github.com/zhangdft/Undergrad-AI4Battery-Projects/issues)
- 讨论区：[Discussions](https://github.com/zhangdft/Undergrad-AI4Battery-Projects/discussions)
- 导师邮箱：zhangbao@uestc.edu.cn
