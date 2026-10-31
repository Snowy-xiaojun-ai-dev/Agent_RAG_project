import streamlit as st
import time
from agent_graph import agent_app

st.set_page_config(page_title="企业级 Agentic RAG")

st.title("企业级自反思智能体 (Agentic RAG)")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户提问
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Agent 开始运行
    with st.chat_message("assistant"):
        # 用 st.status 展示思考过程
        with st.status("Agent 正在思考中...", expanded=True) as status:
            st.write("正在检索本地知识库...")
            time.sleep(0.5) # 模拟停顿让动画更自然
            
            # 跑我们的 LangGraph
            result = agent_app.invoke({"question": prompt})
            
            # 获取反思结果
            reflection = result.get("reflection", "")
            if "不能" in reflection:
                st.write("评估结果：本地知识库无法回答。")
                st.write("自动触发 Tavily 联网搜索...")
            else:
                st.write("评估结果：本地知识库可回答。")
                
            st.write("正在生成最终答案...")
            status.update(label="思考完成！", state="complete", expanded=False)
        
        # 打印最终答案
        final_answer = result.get("generation", "抱歉，系统错误。")
        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})