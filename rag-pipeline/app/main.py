from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv


# Load credentials before importing LangGraph or LangChain components.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from src.agents.agent_graph import build_graph


def new_conversation_config():
    return {
        "configurable": {
            "thread_id": str(uuid4()),
        }
    }


def main():
    graph = build_graph(enable_memory=True)
    config = new_conversation_config()

    print("Hospital Information Assistant")
    print("Fictional hospital information for learning.")
    print("Conversation memory is enabled for this session.")
    print("Type 'new' to start a fresh conversation.")
    print("Type 'exit' or 'quit' to close.\n")

    while True:
        try:
            question = input("You: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if question.lower() == "new":
                config = new_conversation_config()
                print(
                    "\nStarted a new conversation. "
                    "Previous messages will not be used.\n"
                )
                continue

            if not question:
                print("Please enter a question.\n")
                continue

            result = graph.invoke(
                {"question": question},
                config=config,
            )

            print("\nAssistant:", result["answer"])

            if result.get("sources"):
                print("\nRetrieved passages — reference key:")

                for source in result["sources"]:
                    print(
                        f"[{source['label']}] {source['filename']} | "
                        f"{source['section']} | "
                        f"{source['subsection']}"
                    )

            print()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()