"""Configure the language model client."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_llm():
    # Find the project's main folder.
    project_root = Path(__file__).resolve().parents[2]

    # Load the API key from the local .env file.
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing. Check your .env file."
        )

    # Configure the model that will generate answers.
    return ChatOpenAI(
        model="gpt-6-luna",
        reasoning_effort="none",
        max_tokens=300,
        timeout=60,
        max_retries=2,
    )


if __name__ == "__main__":
    llm = get_llm()

    # This test makes a paid API request.
    response = llm.invoke(
        "Reply with one short sentence confirming you are ready "
        "to help with a fictional hospital information demo."
    )

    print("Model response:", response.content)