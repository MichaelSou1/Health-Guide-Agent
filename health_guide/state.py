from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    # messages 存储短期记忆，使用 operator.add 进行追加 
    messages: Annotated[List[AnyMessage], operator.add]
    # next 用于 Supervisor 决定下一个执行者是谁
    next: str
