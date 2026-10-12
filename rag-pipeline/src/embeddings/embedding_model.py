"""Configure and create the embedding model."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


def get_embedding_model():
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

    return OpenAIEmbeddings(model="text-embedding-3-small")


if __name__ == "__main__":
    embedding_model = get_embedding_model()

    sample_text = "The cardiology department is in the East Tower."

    vector = embedding_model.embed_query(sample_text)

    print("Sample text:", sample_text)
    print("Number of dimensions:", len(vector))
    print("First 5 numbers:", vector[:5])
    print("Embedding created successfully!")