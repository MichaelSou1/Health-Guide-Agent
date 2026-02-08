import os
from langchain_core.tools import tool

import requests
import json

# 1. 定义秘塔搜索工具 (Metaso Search)
@tool
def metaso_search(query: str):
    """当需要获取互联网上的最新信息、健康知识或解决特定问题时使用此工具。"""
    
    url = "https://api.ecn.ai/metaso/search"
    api_key = os.environ.get('METASO_API_KEY')
    
    if not api_key:
        return "[System Error] METASO_API_KEY not found in environment variables."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "q": query,
        "scope": "webpage",
        "includeSummary": True,
        "size": 5 
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # 提取关键信息，避免 Context 过长
        results = []
        if "podcasts" in data: # 文档中返回的字段示例是 podcasts，但通常 search 结果可能是列表
             for item in data.get("podcasts", []):
                title = item.get("title", "No Title")
                snippet = item.get("snippet", "No Snippet")
                link = item.get("link", "#")
                results.append(f"- [{title}]({link}): {snippet}")
        
        # 如果返回结构不同（文档示例可能不完整），尝试一种通用的提取方式
        if not results and isinstance(data, list):
             for item in data:
                results.append(str(item))
        
        if not results:
            return f"Search executed but returned no clear results. Raw response keys: {list(data.keys())}"

        return "\n".join(results)

    except Exception as e:
        return f"Metaso Search failed: {e}"

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
tools = [metaso_search, calculate_tdee]
