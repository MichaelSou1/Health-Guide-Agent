from typing import TypedDict, Annotated, List, Dict
import operator
from langchain_core.messages import AnyMessage


def _merge_expert_responses(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    return {**a, **b}


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    # 本轮路由的专家列表（支持多专家并行）
    next: List[str]
    profile_user_id: str
    # 并行执行时各专家写入，用 operator.add 合并
    last_tools: Annotated[List[str], operator.add]
    retrieval_hits: Annotated[int, operator.add]
    # 各专家本轮回答，key=专家名，value=回答文本
    expert_responses: Annotated[Dict[str, str], _merge_expert_responses]
