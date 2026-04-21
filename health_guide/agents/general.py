from ..tools import get_user_profile, update_user_profile, retrieve_general_knowledge
from ..utils import create_agent
from ..llm import llm
from ..profile_store import get_user_profile as get_profile_from_store, profile_to_prompt_text

def _build_general_agent(user_id: str):
    profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    system_prompt = (
        "你是健康管理团队的贴心助理。"
        f"当前用户画像：{profile_text}。"
        "负责寒暄、通用问题与多轮澄清。"
        "若用户询问健康常识并需要给出建议，先调用一次 retrieve_general_knowledge 再回答。"
        "若用户问到基础健康常识，可调用 retrieve_general_knowledge 检索共享与通用语料。"
        "若用户提供新的个人偏好、作息、目标变化，请调用 update_user_profile 进行记录。"
        "回答保持自然、简洁。"
        "优先短回复，不主动给健康方案；只有用户明确询问时再进入健康建议。"
    )
    return create_agent(llm, [retrieve_general_knowledge, get_user_profile, update_user_profile], system_prompt)

def general_node(state):
    user_id = state.get("profile_user_id", "default_user")
    general_agent = _build_general_agent(user_id)
    result = general_agent.invoke({"messages": state["messages"]})
    used_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                used_tools.append(call.get("name", "Unknown"))

    retrieval_hits = sum(1 for t in used_tools if "retrieve" in t and "knowledge" in t)
    return {
        "expert_responses": {"General": result["messages"][-1].content},
        "last_tools": used_tools,
        "retrieval_hits": retrieval_hits,
    }
