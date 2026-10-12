"""Retrieve relevant hospital document chunks."""
from src.vector_store.vector_db import get_vector_store


def retrieve_documents(question, k=6):
    vector_store = get_vector_store()

    return vector_store.similarity_search(
        query=question,
        k=k,
    )


if __name__ == "__main__":
    question = "Where is the cardiology department?"

    documents = retrieve_documents(question)

    print("Question:", question)
    print("Results found:", len(documents))

    for number, document in enumerate(documents, start=1):
        print(f"\n--- Result {number} ---")
        print("Source:", document.metadata.get("source"))
        print("Section:", document.metadata.get("section"))
        print(document.page_content)