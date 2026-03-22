from langchain_core.messages import SystemMessage
from ..llm import llm


def _rule_based_route(user_text: str) -> str:
    text = (user_text or "").lower()

    # 训练后酸痛/恢复优先归 Trainer（用户期望：训练相关恢复由教练承接）
    trainer_keywords = [
        "健身", "训练", "练腿", "练胸", "练背", "跑步", "运动后", "运动", "肌肉酸痛", "doms", "拉伤", "动作", "组数", "强度",
    ]
    wellness_keywords = ["失眠", "焦虑", "压力", "情绪", "冥想", "放松", "心理", "睡眠"]

    if any(k in text for k in trainer_keywords):
        return "Trainer"
    if any(k in text for k in wellness_keywords):
        return "Wellness"
    return ""

# --- 3. 定义 Supervisor (总监) ---
# 总监负责输出 JSON 格式的路由指令
supervisor_system_prompt = (
    "你是健康管理团队的总监。根据用户输入，决定下一步由谁处理。"
    "可选角色: ['Trainer', 'Nutritionist', 'Wellness', 'General']。"
    "训练计划、动作安排、体能提升 -> Trainer；"
    "运动后肌肉酸痛、训练相关恢复、训练损伤风险评估 -> Trainer；"
    "饮食热量、营养搭配、食材替换 -> Nutritionist；"
    "非训练场景的睡眠压力、情绪调节、心理恢复 -> Wellness；"
    "寒暄和泛化问题 -> General。"
    "如果用户是在寒暄、打招呼或闲聊，请选择 'General'。"
    "如果任务已完成或需要用户回答，请选择 'FINISH'。"
    "直接输出角色名称，不要输出其他废话。"
)

def supervisor_node(state):
    messages = state['messages']
    user_text = ""
    if messages:
        user_text = str(getattr(messages[-1], "content", ""))

    forced_route = _rule_based_route(user_text)
    if forced_route:
        return {"next": forced_route}

    # 简单的路由逻辑：让 LLM 决定下一个是谁
    response = llm.invoke([SystemMessage(content=supervisor_system_prompt)] + messages)
    next_role = response.content.strip().replace("'", "").replace('"', "")
    
    # 容错处理
    if next_role not in ['Trainer', 'Nutritionist', 'Wellness', 'General', 'FINISH']:
        next_role = 'FINISH'
        
    return {"next": next_role}
