"""
命令行交互模式
方便本地快速测试 Agent，无需启动 API 服务

用法:
    python main.py
"""

from langchain_core.messages import HumanMessage
from app.graph import build_graph


def main():
    print("=" * 50)
    print("🤖 IT Helpdesk Agent - 命令行测试模式")
    print("=" * 50)
    print("输入你的问题，输入 'quit' 或 'exit' 退出\n")

    agent = build_graph(with_memory=True)
    thread_id = "cli-test-001"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        try:
            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_id": "EMP001",
                    "user_name": "测试用户",
                    "user_department": "工程部",
                    "needs_human": False,
                    "confidence": 1.0,
                    "intent": "",
                    "retrieved_context": "",
                    "ticket_info": {},
                },
                config=config,
            )

            # 提取最后一条 AI 消息
            reply = result["messages"][-1].content
            intent = result.get("intent", "")

            print(f"\n🤖 Agent [{intent}]: {reply}\n")

        except Exception as e:
            print(f"\n❌ 出错了: {e}\n")


if __name__ == "__main__":
    main()
