import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Send

from .state import AgentState
from .agents.supervisor import supervisor_node
from .agents.trainer import trainer_node
from .agents.nutritionist import nutritionist_node
from .agents.wellness import wellness_node
from .agents.general import general_node
from .agents.aggregator import aggregator_node

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

_VALID_EXPERTS = {"Trainer", "Nutritionist", "Wellness", "General"}


def _route_to_experts(state: AgentState):
    """并行 fan-out：向每个被选中的专家发送 Send，LangGraph 会并行执行。"""
    experts = state.get("next", [])
    targets = [e for e in experts if e in _VALID_EXPERTS]
    if not targets:
        return END
    return [Send(expert, state) for expert in targets]


workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Trainer", trainer_node)
workflow.add_node("Nutritionist", nutritionist_node)
workflow.add_node("Wellness", wellness_node)
workflow.add_node("General", general_node)
workflow.add_node("Aggregator", aggregator_node)

# Supervisor → 并行 fan-out 给各专家（或直接 END）
workflow.add_conditional_edges(
    "Supervisor",
    _route_to_experts,
    ["Trainer", "Nutritionist", "Wellness", "General", END],
)

# 所有专家完成后汇聚到 Aggregator（LangGraph 自动等待所有并行分支）
workflow.add_edge("Trainer", "Aggregator")
workflow.add_edge("Nutritionist", "Aggregator")
workflow.add_edge("Wellness", "Aggregator")
workflow.add_edge("General", "Aggregator")
workflow.add_edge("Aggregator", END)

workflow.set_entry_point("Supervisor")

graph = workflow.compile(checkpointer=memory)
