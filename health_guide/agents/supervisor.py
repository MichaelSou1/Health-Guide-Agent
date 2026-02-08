from langchain_core.messages import SystemMessage
from ..llm import llm

# --- 3. 定义 Supervisor (总监) ---
# 总监负责输出 JSON 格式的路由指令
supervisor_system_prompt = (
    "你是健康管理团队的总监。根据用户输入，决定下一步由谁处理。"
    "可选角色: ['Trainer', 'Nutritionist', 'Wellness', 'General']。"
    "如果用户是在寒暄、打招呼或闲聊，请选择 'General'。"
    "如果任务已完成或需要用户回答，请选择 'FINISH'。"
    "直接输出角色名称，不要输出其他废话。"
)

def supervisor_node(state):
    messages = state['messages']
    # 简单的路由逻辑：让 LLM 决定下一个是谁
    response = llm.invoke([SystemMessage(content=supervisor_system_prompt)] + messages)
    next_role = response.content.strip().replace("'", "").replace('"', "")
    
    # 容错处理
    if next_role not in ['Trainer', 'Nutritionist', 'Wellness', 'General']:
        next_role = 'FINISH'
        
    return {"next": next_role}
