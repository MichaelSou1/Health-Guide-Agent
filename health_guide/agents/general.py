import json
from ..utils import create_agent
from ..llm import llm
from ..config import USER_PROFILE

# 综合助手：负责日常闲聊、寒暄以及不属于特定领域的通用问题
# 注意：json.dumps 产生的大括号会被 LangChain 认为是 Prompt 变量，需要转义
user_profile_str = json.dumps(USER_PROFILE).replace("{", "{{").replace("}", "}}")

general_agent = create_agent(llm, [], 
    f"你是健康管理团队的贴心助理。用户画像：{user_profile_str}。"
    "你的职责是与用户进行日常寒暄、回应问候，或者处理不属于训练、饮食、康复领域的通用问题。"
    "请语气亲切、自然。")

def general_node(state):
    result = general_agent.invoke(state)
    return {"messages": [result]}
