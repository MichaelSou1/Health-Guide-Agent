from ..tools import retrieve_wellness_knowledge, get_user_profile, update_user_profile
from ..utils import create_agent
from ..llm import extract_text_content, llm
from ..profile_store import get_user_profile as get_profile_from_store, profile_to_prompt_text
from ._scratchpad import format_peer_notes, build_scratchpad_note

def _build_wellness_agent(user_id: str, peer_notes_text: str):
    profile_text = profile_to_prompt_text(get_profile_from_store(user_id))
    system_prompt = (
        "你是身心康复师。"
        f"当前用户画像：{profile_text}。"
        f"{peer_notes_text}"
        "当用户询问康复/睡眠/压力等建议时，必须先调用一次 retrieve_wellness_knowledge 再回答。"
        "若 retrieve_wellness_knowledge 返回了知识片段（结果包含'命中以下知识片段'），"
        "必须直接基于这些片段作答，不得仅凭模型内部知识回答。"
        "仅当 retrieve_wellness_knowledge 明确返回'未命中本地知识库'时，"
        "才可不再依赖知识库片段，直接用你的通用康复、睡眠与压力管理知识给出保守兜底建议。"
        "若用户透露新的压力来源、睡眠信息、疼痛变化，请调用 update_user_profile 记录。"
        "输出时兼顾心理支持、恢复节奏与风险边界。"
        "【个性化要求】回答必须结合画像中已知的压力来源（mental_state.stress_sources）、"
        "睡眠状态和身份背景，给出有针对性的建议而非通用清单。"
        "例如：若画像显示压力来源是'工作加班'，建议应结合该场景（如碎片化放松、下班边界管理），"
        "而不是泛泛说'建议减少压力'。"
        "若用户描述了身体症状（持续疼痛、心率异常、长期失眠超过两周等），"
        "必须首先建议就医，再提供辅助性建议。"
    )
    return create_agent(
        llm,
        [retrieve_wellness_knowledge, get_user_profile, update_user_profile],
        system_prompt,
    )

def wellness_node(state):
    user_id = state.get("profile_user_id", "default_user")
    peer_notes_text = format_peer_notes(state.get("agent_notes") or {}, self_role="Wellness")
    wellness_agent = _build_wellness_agent(user_id, peer_notes_text)
    result = wellness_agent.invoke({"messages": state["messages"]})
    used_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                used_tools.append(call.get("name", "Unknown"))

    retrieval_hits = sum(1 for t in used_tools if "retrieve" in t and "knowledge" in t)
    answer = extract_text_content(result["messages"][-1])
    return {
        "expert_responses": {"Wellness": answer},
        "agent_notes": {"Wellness": build_scratchpad_note("Wellness", answer)},
        "last_tools": used_tools,
        "retrieval_hits": retrieval_hits,
    }
