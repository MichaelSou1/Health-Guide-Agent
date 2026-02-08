import json
from ..tools import calculate_tdee
from ..utils import create_agent
from ..llm import llm
from ..config import USER_PROFILE

# 教练：负责训练计划 (使用 TDEE 工具辅助判断强度)
user_profile_str = json.dumps(USER_PROFILE).replace("{", "{{").replace("}", "}}")
trainer_agent = create_agent(llm, [calculate_tdee], 
    f"你是力量训练教练。用户画像：{user_profile_str}。注意用户的伤病历史。")

def trainer_node(state):
    result = trainer_agent.invoke(state)
    return {"messages": [result]}
