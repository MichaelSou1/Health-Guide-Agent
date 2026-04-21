from ..tools import metaso_search, retrieve_nutritionist_knowledge, get_user_profile, update_user_profile
from ..utils import create_agent
from ..llm import llm
from ..profile_store import get_user_profile as get_profile_from_store, profile_to_prompt_text

def _build_nutritionist_agent(user_id: str):
    profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    system_prompt = (
        "你是膳食营养师。"
        f"当前用户画像：{profile_text}。"
        "只要用户在问饮食/营养建议，必须先调用一次 retrieve_nutritionist_knowledge 再回答。"
        "优先调用 retrieve_nutritionist_knowledge 获取本地可追溯知识；"
        "需要最新外部信息时再调用 metaso_search。"
        "如果用户补充了口味偏好/禁忌/目标变化，请调用 update_user_profile 更新画像。"
        "输出请给出清晰饮食方案（热量、三大营养素、可替代食材）。"
    )
    return create_agent(
        llm,
        [retrieve_nutritionist_knowledge, metaso_search, get_user_profile, update_user_profile],
        system_prompt,
    )

def nutritionist_node(state):
    user_id = state.get("profile_user_id", "default_user")
    nutritionist_agent = _build_nutritionist_agent(user_id)
    result = nutritionist_agent.invoke({"messages": state["messages"]})
    used_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                used_tools.append(call.get("name", "Unknown"))

    retrieval_hits = sum(1 for t in used_tools if "retrieve" in t and "knowledge" in t)
    return {
        "expert_responses": {"Nutritionist": result["messages"][-1].content},
        "last_tools": used_tools,
        "retrieval_hits": retrieval_hits,
    }
