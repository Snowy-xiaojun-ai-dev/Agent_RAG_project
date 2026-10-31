# 企业级自反思智能体 (Agentic RAG)

本项目基于 **LangGraph** 与 **LlamaIndex** 构建，突破了传统 RAG（检索增强生成）的线性流程，引入了**智能体（Agent）架构**。系统具备自主思考、自我反思与联网搜索能力，并集成了 RAGAs 量化评估体系。

## 核心特性

*   **自主反思机制 (Self-Reflection)**：大模型在检索后会自主评估上下文质量。若本地知识库无法回答，将自动触发 `Tavily` 外部搜索引擎进行实时数据获取。
*   **量化评估指标 (RAGAs)**：引入业界标准评估框架，对系统进行客观打分。
    *   *Faithfulness (忠实度)*: **0.6667**
    *   *Answer Relevance (相关性)*: **0.8052**
*   **透明化推理**：使用 Streamlit 构建前端，完整展示 Agent 的思维链 (Chain of Thought)，让“黑盒”可视化。
*   **安全工程化**：使用 `python-dotenv` 进行密钥隔离，防止 API Key 泄漏。

## 系统架构

`用户提问` -> `本地知识库检索` -> `[大模型自我反思]`
*   若命中 -> `生成最终答案`
*   若未命中 -> `触发 Tavily 联网搜索` -> `生成最终答案`

## 快速启动

1. 安装依赖：
   ```bash
   pip install langgraph langchain-openai tavily-python ragas streamlit python-dotenv langchain-huggingface
