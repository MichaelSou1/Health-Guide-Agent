import os
from langchain_openai import ChatOpenAI
from . import config # Ensure .env is loaded

model_name = os.environ.get("SILICONFLOW_MODEL")
if not model_name:
    raise ValueError("SILICONFLOW_MODEL environment variable is not set. Please set it in .env file.")

llm = ChatOpenAI(
    model=model_name,
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.environ.get("SILICONFLOW_API_KEY")
)
