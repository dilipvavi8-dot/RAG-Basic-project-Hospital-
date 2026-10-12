"""Build and persist the hospital document index."""
import hashlib
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.document_loader import load_document
from src.processing.chunking import split_into_chunks
from src.vector_store.vector_db import get_vector_store


def build_index():
    project_root = Path(__file__).resolve().parents[1]
    document_path = project_root / "data" / "raw" / "hospital_info.txt"

    document = load_document(document_path)
    chunks = split_into_chunks(document)

    if not chunks:
        raise ValueError("No chunks found. Check your hospital document.")

    vector_store = get_vector_store()

    existing = vector_store.get(
        where={"source": document["source"]},
        include=[],
    )
    existing_ids = set(existing["ids"])

    current_ids = set()
    new_documents = []
    new_ids = []

    for chunk in chunks:
        identity = (
            f"{chunk['source']}:{chunk['chunk_id']}:{chunk['text']}"
        )
        record_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        current_ids.add(record_id)

        if record_id in existing_ids:
            continue

        new_documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "section": chunk.get("section", ""),
                    "subsection": chunk.get("subsection", ""),
                },
            )
        )
        new_ids.append(record_id)

    print("Total chunks:", len(chunks))
    print("Chunks needing embeddings:", len(new_documents))

    if new_documents:
        vector_store.add_documents(
            documents=new_documents,
            ids=new_ids,
        )

    # Remove outdated records only after new records are saved.
    outdated_ids = list(existing_ids - current_ids)
    if outdated_ids:
        vector_store.delete(ids=outdated_ids)

    print("Outdated records removed:", len(outdated_ids))
    print("Indexing complete!")


if __name__ == "__main__":
    build_index()