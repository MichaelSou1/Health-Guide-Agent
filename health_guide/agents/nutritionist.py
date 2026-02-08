import json
from ..tools import metaso_search
from ..utils import create_agent
from ..llm import llm
from ..config import USER_PROFILE

# 营养师：负责饮食建议 (使用 Search 工具查询食物热量)
user_profile_str = json.dumps(USER_PROFILE).replace("{", "{{").replace("}", "}}")
nutritionist_agent = create_agent(llm, [metaso_search], 
    f"你是膳食营养师。用户画像：{user_profile_str}。请结合用户家庭饮食习惯给出建议。")

def nutritionist_node(state):
    result = nutritionist_agent.invoke(state)
    return {"messages": [result]}
