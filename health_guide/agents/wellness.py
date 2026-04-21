from ..tools import metaso_search, retrieve_wellness_knowledge, get_user_profile, update_user_profile
from ..utils import create_agent
from ..llm import llm
from ..profile_store import get_user_profile as get_profile_from_store, profile_to_prompt_text

def _build_wellness_agent(user_id: str):
    profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    system_prompt = (
        "你是身心康复师。"
        f"当前用户画像：{profile_text}。"
        "当用户询问康复/睡眠/压力等建议时，必须先调用一次 retrieve_wellness_knowledge 再回答。"
        "若 retrieve_wellness_knowledge 返回了知识片段（结果包含"命中以下知识片段"），"
        "必须直接基于这些片段作答，不得再调用 metaso_search，也不得仅凭模型内部知识回答。"
        "仅当 retrieve_wellness_knowledge 明确返回"未命中本地知识库"时，"
        "才可将 metaso_search 作为兜底补充。"
        "若用户透露新的压力来源、睡眠信息、疼痛变化，请调用 update_user_profile 记录。"
        "输出时兼顾心理支持、恢复节奏与风险边界。"
    )
    return create_agent(
        llm,
        [retrieve_wellness_knowledge, metaso_search, get_user_profile, update_user_profile],
        system_prompt,
    )

def wellness_node(state):
    user_id = state.get("profile_user_id", "default_user")
    wellness_agent = _build_wellness_agent(user_id)
    result = wellness_agent.invoke({"messages": state["messages"]})
    used_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                used_tools.append(call.get("name", "Unknown"))

    retrieval_hits = sum(1 for t in used_tools if "retrieve" in t and "knowledge" in t)
    return {
        "expert_responses": {"Wellness": result["messages"][-1].content},
        "last_tools": used_tools,
        "retrieval_hits": retrieval_hits,
    }
