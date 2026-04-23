import os
import re
from langchain_core.tools import tool

import json

from .rag import get_kb
from .profile_store import get_user_profile as get_profile_from_store
from .profile_store import update_user_profile as update_profile_in_store

def _retrieve_by_agent(query: str, top_k: int, agent: str) -> str:
    kb = get_kb()
    results = kb.retrieve(query=query, top_k=top_k, agent=agent)
    if not results:
        return "[RAG] 未命中本地知识库，请尝试改写查询或补充 knowledge_base 文档。"

    lines = ["[RAG] 命中以下知识片段："]
    for i, r in enumerate(results, start=1):
        snippet = re.sub(r"\s+", " ", r["content"]).strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        lines.append(
            f"{i}. [source: {r['source']} | chunk: {r['chunk_id']} | ns: {r.get('namespace', 'n/a')} | score: {r['score']}] {snippet}"
        )
    return "\n".join(lines)


@tool
def retrieve_health_knowledge(query: str, top_k: int = 4, agent: str = "general"):
    """通用检索工具：从分层知识库检索（shared + 指定 agent 私有语料）。"""
    return _retrieve_by_agent(query=query, top_k=top_k, agent=agent)


@tool
def retrieve_trainer_knowledge(query: str, top_k: int = 4):
    """训练教练专用：优先检索 trainer 私有知识库，再补充 shared。"""
    return _retrieve_by_agent(query=query, top_k=top_k, agent="trainer")


@tool
def retrieve_nutritionist_knowledge(query: str, top_k: int = 4):
    """营养师专用：优先检索 nutritionist 私有知识库，再补充 shared。"""
    return _retrieve_by_agent(query=query, top_k=top_k, agent="nutritionist")


@tool
def retrieve_wellness_knowledge(query: str, top_k: int = 4):
    """康复师专用：优先检索 wellness 私有知识库，再补充 shared。"""
    return _retrieve_by_agent(query=query, top_k=top_k, agent="wellness")


@tool
def retrieve_general_knowledge(query: str, top_k: int = 4):
    """通用助手专用：优先检索 general 私有知识库，再补充 shared。"""
    return _retrieve_by_agent(query=query, top_k=top_k, agent="general")


@tool
def get_user_profile(user_id: str = ""):
    """获取用户画像。user_id 为空时，将使用环境变量 HEALTH_GUIDE_USER_ID。"""
    target_user_id = user_id or os.environ.get("HEALTH_GUIDE_USER_ID", "default_user")
    profile = get_profile_from_store(target_user_id)
    return json.dumps(profile, ensure_ascii=False)


@tool
def update_user_profile(patch_json: str, user_id: str = ""):
    """更新用户画像。patch_json 需是 JSON 字符串，将做深度合并。"""
    target_user_id = user_id or os.environ.get("HEALTH_GUIDE_USER_ID", "default_user")
    try:
        patch = json.loads(patch_json)
        if not isinstance(patch, dict):
            return "[Profile Update Error] patch_json 必须是 JSON 对象。"
    except Exception as e:
        return f"[Profile Update Error] 无法解析 JSON: {e}"

    updated = update_profile_in_store(target_user_id, patch)
    return f"用户画像已更新：{json.dumps(updated, ensure_ascii=False)}"

# 2. 定义 TDEE 计算工具
@tool
def calculate_tdee(weight_kg: float, height_cm: float, age: int, activity_level: str = "sedentary"):
    """根据体重、身高、年龄计算每日热量消耗(TDEE)。activity_level 可选: sedentary, active, very_active"""
    # Mifflin-St Jeor 公式
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    
    multipliers = {
        "sedentary": 1.2,
        "active": 1.55,
        "very_active": 1.725
    }
    tdee = bmr * multipliers.get(activity_level, 1.2)
    return f"根据公式计算，基础代谢(BMR)为 {bmr} kcal，每日总消耗(TDEE)约为 {int(tdee)} kcal。"

# 工具列表
tools = [
    retrieve_health_knowledge,
    retrieve_trainer_knowledge,
    retrieve_nutritionist_knowledge,
    retrieve_wellness_knowledge,
    retrieve_general_knowledge,
    calculate_tdee,
    get_user_profile,
    update_user_profile,
]
