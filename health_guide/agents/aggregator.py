from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from ..llm import llm

_EXPERT_DISPLAY_NAMES = {
    "Trainer": "力量训练教练",
    "Nutritionist": "膳食营养师",
    "Wellness": "身心康复师",
    "General": "健康助理",
}


def aggregator_node(state):
    current_experts = state.get("next", [])
    all_responses = state.get("expert_responses", {})
    # 只取本轮被调用的专家回答，过滤掉历史轮次的残留
    relevant = {k: v for k, v in all_responses.items() if k in current_experts}

    if not relevant:
        return {"messages": [AIMessage(content="抱歉，未能获取专家建议，请重试。")]}

    if len(relevant) == 1:
        content = next(iter(relevant.values()))
        return {"messages": [AIMessage(content=content)]}

    # 多专家：让 LLM 整合成一份连贯建议
    parts = []
    for expert_key, response in relevant.items():
        display_name = _EXPERT_DISPLAY_NAMES.get(expert_key, expert_key)
        parts.append(f"【{display_name}】\n{response}")
    combined = "\n\n---\n\n".join(parts)

    synthesis_prompt = (
        f"以下是多位专家对用户问题的各自建议：\n\n{combined}\n\n"
        "请将以上建议整合为一份连贯、全面的综合健康方案。"
        "要求：避免重复内容；突出各专家的专业视角；给出清晰可执行的建议；保持自然流畅。"
    )

    response = llm.invoke([
        SystemMessage(content="你是综合健康顾问，负责将多位专家的建议整合为统一的最终方案。"),
        HumanMessage(content=synthesis_prompt),
    ])

    return {"messages": [AIMessage(content=response.content)]}
