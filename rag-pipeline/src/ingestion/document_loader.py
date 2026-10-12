"""Load hospital documents while retaining source metadata."""
from pathlib import Path


def load_document(file_path):
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    return {
        "text": text,
        "source": path.name,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    document_path = project_root / "data" / "raw" / "hospital_info.txt"

    document = load_document(document_path)

    print("Source:", document["source"])
    print("\nHospital document:\n")
    print(document["text"])