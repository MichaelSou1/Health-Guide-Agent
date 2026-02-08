import json
from ..tools import metaso_search
from ..utils import create_agent
from ..llm import llm
from ..config import USER_PROFILE

# 康复/心理师：负责压力和恢复 (使用 Search 工具查询康复知识)
user_profile_str = json.dumps(USER_PROFILE).replace("{", "{{").replace("}", "}}")
wellness_agent = create_agent(llm, [metaso_search], 
    f"你是身心康复师。用户画像：{user_profile_str}。关注用户的压力来源和身体疼痛。")

def wellness_node(state):
    result = wellness_agent.invoke(state)
    return {"messages": [result]}
