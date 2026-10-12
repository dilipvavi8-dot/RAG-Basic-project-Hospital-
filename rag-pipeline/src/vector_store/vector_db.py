"""Create, persist, and open the Chroma vector index."""
from pathlib import Path

from langchain_chroma import Chroma
from src.embeddings.embedding_model import get_embedding_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "hospital_info"


def get_vector_store():
    embedding_model = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(DATABASE_PATH),
    )


if __name__ == "__main__":
    vector_store = get_vector_store()

    print("Chroma database is ready!")
    print("Database location:", DATABASE_PATH)
    print("Collection:", COLLECTION_NAME)