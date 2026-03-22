import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import AgentState
from .agents.supervisor import supervisor_node
from .agents.trainer import trainer_node
from .agents.nutritionist import nutritionist_node
from .agents.wellness import wellness_node
from .agents.general import general_node

# 使用本地文件作为 Checkpoint
# 注意：在生产环境中，通常会在 application context 中管理连接
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Trainer", trainer_node)
workflow.add_node("Nutritionist", nutritionist_node)
workflow.add_node("Wellness", wellness_node)
workflow.add_node("General", general_node)

# 定义路由逻辑
workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"], # 根据 state['next'] 决定去向
    {
        "Trainer": "Trainer",
        "Nutritionist": "Nutritionist",
        "Wellness": "Wellness",
        "General": "General",
        "FINISH": END
    }
)

# 专家执行完后，必须汇报给 Supervisor (形成闭环)
workflow.add_edge("Trainer", "Supervisor")
workflow.add_edge("Nutritionist", "Supervisor")
workflow.add_edge("Wellness", "Supervisor")
workflow.add_edge("General", "Supervisor")

# 设置入口
workflow.set_entry_point("Supervisor")

# 编译图
graph = workflow.compile(checkpointer=memory)
