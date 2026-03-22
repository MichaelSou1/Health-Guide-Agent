from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    # messages 存储短期记忆，使用 operator.add 进行追加 
    messages: Annotated[List[AnyMessage], operator.add]
    # next 用于 Supervisor 决定下一个执行者是谁
    next: str
    # 当前对话对应的用户 ID（用于个性化画像加载）
    profile_user_id: str
    # 最近一次专家节点调用过的工具名
    last_tools: List[str]
    # 最近一次是否命中 RAG
    retrieval_hits: int
