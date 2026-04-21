from typing import List
from langchain_core.messages import SystemMessage
from ..llm import llm

_VALID_EXPERTS = {"Trainer", "Nutritionist", "Wellness", "General"}


def _rule_based_route(user_text: str) -> List[str]:
    text = (user_text or "").lower()

    trainer_keywords = [
        "健身", "训练", "练腿", "练胸", "练背", "跑步", "运动后", "运动", "肌肉酸痛", "doms", "拉伤", "动作", "组数", "强度",
    ]
    wellness_keywords = ["失眠", "焦虑", "压力", "情绪", "冥想", "放松", "心理", "睡眠"]
    nutrition_keywords = ["饮食", "热量", "营养", "蛋白质", "碳水", "食材", "食谱", "卡路里", "减脂餐"]

    results = []
    if any(k in text for k in trainer_keywords):
        results.append("Trainer")
    if any(k in text for k in nutrition_keywords):
        results.append("Nutritionist")
    if any(k in text for k in wellness_keywords):
        results.append("Wellness")
    return results


supervisor_system_prompt = (
    "你是健康管理团队的总监。根据用户输入，决定需要哪些专家同时处理。\n"
    "可选角色: Trainer, Nutritionist, Wellness, General。\n"
    "可以同时选择多个角色，用英文逗号分隔（不加空格）。\n"
    "路由规则：\n"
    "- 训练计划、动作安排、体能提升、训练后恢复、肌肉酸痛、训练损伤 -> Trainer\n"
    "- 饮食热量、营养搭配、食材替换、减脂增肌饮食 -> Nutritionist\n"
    "- 非训练场景的睡眠、压力、情绪调节、心理恢复 -> Wellness\n"
    "- 寒暄、打招呼、泛化问题 -> General\n"
    "如果问题同时涉及多个领域（如训练+营养、运动+睡眠），请同时选择对应的多个专家。\n"
    "如果任务已完成或需要用户回答，请输出: FINISH\n"
    "直接输出角色名称，多个时用英文逗号分隔，不要输出其他内容。\n"
    "示例: Trainer 或 Trainer,Nutritionist 或 FINISH"
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
