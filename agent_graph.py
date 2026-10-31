import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. 加载 .env 保险箱
load_dotenv()

# ================= 1. 配置大模型与工具 =================
# 从保险箱中获取智谱 API Key
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 初始化智谱 GLM-4 模型 (兼容 OpenAI 接口)
llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPU_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1
)

# 配置 Tavily 联网搜索工具 (从保险箱中获取 TAVILY_API_KEY)
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
web_search_tool = TavilySearchResults(max_results=2)

# 读取本地知识库
with open("data.txt", "r", encoding="utf-8") as f:
    LOCAL_DOCS = f.read()

# ================= 2. 定义状态字典 =================
class AgentState(TypedDict):
    question: str        # 用户提问
    documents: List[str] # 检索到的文档内容
    reflection: str      # 自我反思的结果
    generation: str      # 最终生成的答案

# ================= 3. 定义节点函数 =================
def retrieve_node(state: AgentState):
    """节点1：本地检索（这里为了演示，直接读取整个文档。真实场景用向量库）"""
    print("\n[Agent 思考中...] 🔍 正在检索本地知识库...")
    return {"documents": [LOCAL_DOCS]}

def reflect_node(state: AgentState):
    """节点2：自我反思（判断检索的内容能不能回答问题）"""
    print("[Agent 思考中...] 🤔 正在评估检索结果质量...")
    question = state["question"]
    docs = "\n".join(state["documents"])
    
    prompt = f"""你是一个严苛的评估员。请判断以下文档能否回答用户的问题。
文档：{docs}
问题：{question}
请只回答"能"或"不能"，不要输出任何其他字符。"""
    
    reflection = llm.invoke(prompt).content.strip()
    print(f"[Agent 思考结果]：当前文档能否回答问题？ -> {reflection}")
    return {"reflection": reflection}

def web_search_node(state: AgentState):
    """节点3：联网搜索（本地无法回答时触发）"""
    print("[Agent 思考中...] ⚠️ 本地无相关知识，正在调用 Tavily 联网搜索...")
    question = state["question"]
    search_results = web_search_tool.invoke(question)
    web_text = "\n".join([res["content"] for res in search_results])
    return {"documents": [web_text]}

def generate_node(state: AgentState):
    """节点4：生成最终答案"""
    print("[Agent 思考中...] 📝 正在生成最终答案...")
    question = state["question"]
    docs = "\n".join(state["documents"])
    
    prompt = f"""请根据以下参考资料，用专业的中文回答用户的问题。
如果参考资料里没有答案，请直接说不知道，不要编造。
参考资料：{docs}
问题：{question}
回答："""
    
    generation = llm.invoke(prompt).content
    return {"generation": generation}

# ================= 4. 构建并编译工作流 =================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("reflect", reflect_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

# 设置起点
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "reflect")

# 设置条件分支 (核心逻辑：根据反思结果决定下一步走向)
workflow.add_conditional_edges(
    "reflect",
    # 如果回答包含“不能”，去联网；否则去生成
    lambda state: "web_search" if "不能" in state["reflection"] else "generate",
    {
        "generate": "generate",
        "web_search": "web_search"
    }
)

# 如果是联网搜索了，搜完也得去生成答案
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# 编译成可执行的应用
agent_app = workflow.compile()

# ================= 5. 测试运行 =================
if __name__ == "__main__":
    print("="*50)
    # 测试 1：本地知识库能回答的问题
    q1 = "我想请年假，怎么申请？"
    print(f"🚀 用户提问：{q1}")
    res1 = agent_app.invoke({"question": q1})
    print(f"🎯 最终回答：{res1['generation']}\n")
    
    print("="*50)
    # 测试 2：本地知识库无法回答，需要触发 Agent 联网的问题
    q2 = "今天广州的天气怎么样？"
    print(f"🚀 用户提问：{q2}")
    res2 = agent_app.invoke({"question": q2})
    print(f"🎯 最终回答：{res2['generation']}\n")