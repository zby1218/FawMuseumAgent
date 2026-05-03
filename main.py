from faw_museum.agent import MuseumAgent


def main() -> None:
    agent = MuseumAgent()
    print("博物馆讲解机器人（输入 q 退出）\n")
    while True:
        user_input = input(">>> ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            break
        if not user_input:
            continue
        result = agent.process(user_input)
        print(f"意图: {result['intent']}")
        print(f"回复: {result['response']}\n")


if __name__ == "__main__":
    main()