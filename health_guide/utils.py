from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 辅助函数：创建一个专家 Agent 节点 ---
def create_agent(llm, tools, system_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm
    agent = prompt | llm_with_tools
    return agent
