from pathlib import Path
from src.ingestion.document_loader import load_document


def split_into_chunks(document):
    chunks = []
    title = ""
    section = ""
    subsection = ""
    content = []

    def save_chunk():
        body = "\n".join(content).strip()

        if not body:
            return

        headings = [
            heading
            for heading in [title, section, subsection]
            if heading
        ]

        text = "\n\n".join(headings + [body])

        chunks.append({
            "text": text,
            "source": document["source"],
            "chunk_id": len(chunks) + 1,
            "section": section,
            "subsection": subsection,
        })

    for line in document["text"].splitlines():
        stripped = line.strip()

        if stripped == "---":
            continue

        if stripped.startswith("# "):
            save_chunk()
            content = []
            title = stripped[2:].strip()
            section = ""
            subsection = ""

        elif stripped.startswith("## "):
            save_chunk()
            content = []
            section = stripped[3:].strip()
            subsection = ""

        elif stripped.startswith("### "):
            save_chunk()
            content = []
            subsection = stripped[4:].strip()

        else:
            content.append(line)

    save_chunk()

    return chunks


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    document_path = project_root / "data" / "raw" / "hospital_info.txt"

    document = load_document(document_path)
    chunks = split_into_chunks(document)

    print("Total chunks:", len(chunks))

    for chunk in chunks:
        print("\n--- Chunk", chunk["chunk_id"], "---")
        print("Source:", chunk["source"])
        print(chunk["text"])