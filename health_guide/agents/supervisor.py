from typing import List
from langchain_core.messages import SystemMessage
from ..llm import llm

_VALID_EXPERTS = {"Trainer", "Nutritionist", "Wellness", "General"}


def _rule_based_route(user_text: str) -> List[str]:
    text = (user_text or "").lower()

    trainer_keywords = [
        "健身", "训练", "练腿", "练胸", "练背", "跑步", "运动后", "运动", "肌肉酸痛", "doms", "拉伤", "动作", "组数", "强度",
    ]
    wellness_keywords = ["失眠", "睡不着", "入睡", "睡眠", "焦虑", "压力", "情绪", "冥想", "放松", "心理"]
    nutrition_keywords = ["饮食", "热量", "营养", "蛋白质", "碳水", "食材", "食谱", "卡路里", "减脂餐", "补充", "吃什么", "增肌餐"]

    results = []
    if any(k in text for k in trainer_keywords):
        results.append("Trainer")
    if any(k in text for k in nutrition_keywords):
        results.append("Nutritionist")
    if any(k in text for k in wellness_keywords):
        results.append("Wellness")
    return results


supervisor_system_prompt = (
    "你是健康管理团队的总监，负责将用户问题分派给最合适的专家。\n"
    "可选角色: Trainer, Nutritionist, Wellness, General。\n"
    "【默认规则】优先只选 1 个最相关的专家。\n"
    "【多专家条件】仅当问题明确、同时涉及两个不同领域的核心诉求时，才选 2 个专家。\n"
    "  例如：'练完腿该吃什么' -> Trainer,Nutritionist（训练恢复+饮食方案，两者缺一不可）\n"
    "  例如：'训练后睡不好' -> Trainer,Wellness（训练负荷+睡眠恢复，两者缺一不可）\n"
    "  反例：'深蹲动作怎么做' -> Trainer（纯训练问题，不需要 Nutritionist）\n"
    "  反例：'我最近压力很大' -> Wellness（纯情绪/心理，不叫其他人）\n"
    "路由参考：\n"
    "- 训练计划、动作、体能、肌肉酸痛、训练损伤 -> Trainer\n"
    "- 饮食热量、营养搭配、食材替换、增肌减脂饮食 -> Nutritionist\n"
    "- 睡眠、压力、情绪、心理恢复（非训练语境） -> Wellness\n"
    "- 寒暄、打招呼、泛化常识 -> General\n"
    "如果任务已完成或需要用户回答，请输出: FINISH\n"
    "直接输出角色名称，多个时用英文逗号分隔（不加空格），不要输出其他内容。"
)


def supervisor_node(state):
    messages = state["messages"]
    user_text = ""
    if messages:
        user_text = str(getattr(messages[-1], "content", ""))

    forced_routes = _rule_based_route(user_text)
    if forced_routes:
        return {"next": forced_routes}

    response = llm.invoke([SystemMessage(content=supervisor_system_prompt)] + messages)
    content = response.content.strip().replace("'", "").replace('"', "")

    if content == "FINISH":
        return {"next": ["FINISH"]}

    roles = [r.strip() for r in content.split(",")]
    valid_roles = [r for r in roles if r in _VALID_EXPERTS]

    if not valid_roles:
        valid_roles = ["General"]

    return {"next": valid_roles}
