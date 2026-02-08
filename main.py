import uuid
from langchain_core.messages import HumanMessage
from health_guide.graph import graph

def main():
    # 配置 Checkpoint 的 thread_id
    # 使用 UUID 生成唯一的会话 ID，这样每次运行都是新的会话，
    # 或者固定一个 ID 以测试持久化记忆 (由于目前是 :memory:，重启后记忆会丢失)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=== 开始运行健康管理 Agent 团队 ===")
    print("输入 'q' 或 'exit' 退出对话。\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except EOFError:
            break

        if not user_input:
            continue
            
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Goodbye!")
            break

        # 使用 stream 模式运行
        # LangGraph 会自动将新消息追加到历史记录中 (因为 state 中配置了 operator.add)
        for event in graph.stream({"messages": [HumanMessage(content=user_input)]}, config):
            for key, value in event.items():
                print(f"\n[当前节点]: {key}")
                
                if "messages" in value:
                    # 打印最新生成的消息
                    last_msg = value["messages"][-1]
                    print(f"[回复内容]: {last_msg.content}")
                    
                    # 如果有工具调用，打印工具信息
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        tool_name = last_msg.tool_calls[0].get('name', 'Unknown')
                        print(f"[调用工具]: {tool_name}")
                
                if "next" in value:
                    print(f"[路由决策]: -> {value['next']}")

if __name__ == "__main__":
    main()
