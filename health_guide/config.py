import os
import json
from dotenv import load_dotenv

_ = load_dotenv()

# 请确保 .env 文件中有 METASO_API_KEY 和 SILICONFLOW_API_KEY

# 长期记忆：用户画像 (User Profile)
# 请在此处修改为您自己的信息
USER_PROFILE = {
  "name": "User", # [示例] "Michael"
  "identity": "用户", # [示例] "CS研究生"
  "physical_stats": {
    "height": 0, # [示例] 180 (cm)
    "weight": 0, # [示例] 75 (kg)
    "age": 0,    # [示例] 24
    "injuries": [] # [示例] ["膝盖轻微疼痛", "左肩不适"]
  },
  "dietary_context": {
    "provider": "Self", # [示例] "Mother" 或 "外卖"
    "preferences": [], # [示例] ["喜欢吃肉", "不吃香菜"]
    "goal": "健康"     # [示例] "增肌" 或 "减脂"
  },
  "mental_state": {
    "stress_sources": [], # [示例] ["论文Deadline", "工作压力"]
    "relaxation_preference": "" # [示例] "打游戏" 或 "看电影"
  }
}
