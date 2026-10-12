# Hospital Information RAG Project Requirements

Act as my senior AI engineer and beginner-friendly Python tutor. Help me build a Hospital Information RAG project locally, one file at a time, while explaining everything in simple English.

## Project goal

Build an assistant that answers questions using fictional hospital documents, including visiting hours, appointments, and registration requirements. Answers should include sources and acknowledge when information is missing.

Start with basic RAG using LangChain and a local Chroma database. Add LangGraph after the basic pipeline works.

## Current setup

- macOS and VS Code.
- Python 3.14.3 and Git 2.41.0.
- Repository: `https://github.com/dilipvavi8-dot/RAG-Basic-project-Hospital-.git`
- Local folder: `/Users/dilipkumarvavilapalli/Desktop/Experiments/hospital-testing/RAG-Basic-project-Hospital-/`
- `.venv` already exists and has been activated in the user's terminal.
- `.gitignore` already excludes the virtual environment, secrets, Python caches, and `chroma_db/`.
- No application code had been created when this plan was written.

## Target structure

| Folder or file | Responsibility |
|---|---|
| `config/settings.yaml` | Non-secret model, chunking, and retrieval settings |
| `data/raw/hospital_info.txt` | Original fictional hospital document |
| `data/processed/` | Generated intermediate data, only if needed |
| `data/sample_queries/questions.json` | Example questions and expected information |
| `src/ingestion/document_loader.py` | Read documents and retain source metadata |
| `src/processing/chunking.py` | Split documents into smaller sections |
| `src/embeddings/embedding_model.py` | Configure the embedding model |
| `src/vector_store/vector_db.py` | Save and open the Chroma index |
| `src/retrieval/retriever.py` | Find relevant document chunks |
| `src/generation/llm_client.py` | Configure the language model connection |
| `src/generation/rag_chain.py` | Combine retrieved text and the question to generate an answer |
| `src/agents/agent_graph.py` | Add LangGraph workflow and search decisions later |
| `src/evaluation/evaluate.py` | Evaluate retrieval and answers against sample questions |
| `app/main.py` | Simple terminal interface |
| `scripts/build_index.py` | Run document loading, chunking, embedding, and storage |
| `docs/SETUP.md` | Detailed setup and troubleshooting instructions |
| `README.md` | Project overview and quick-start instructions |
| `requirements.txt` | Verified project dependencies |
| `.env.example` | Configuration placeholders without real secrets |
| `.gitignore` | Exclude secrets and generated local files |

Add `__init__.py` files where appropriate. Use consistent imports and commands that run from the repository root.

Keep `.env`, `.venv/`, and `chroma_db/` local and ignored by Git.

Do not add duplicate retrievers, unnecessary wrapper files, Docker, an API, a UI, reranking, or hybrid search at the beginning. Introduce optional features after the core project works.

## Teaching and implementation rules

1. Create or edit only one project file per learning step, including configuration files.
2. Before each change, explain its purpose using a simple everyday example.
3. After the change, explain the important code, including new Python concepts.
4. Give the exact command to run, where to run it, and the expected output.
5. If a file cannot run independently, explain that and provide an appropriate check.
6. Stop after each step and wait for output or “continue.”
7. Fix errors before moving forward.
8. Preserve existing work; do not overwrite files without inspecting them.
9. Install dependencies gradually. Verify current official documentation and Python 3.14 compatibility before choosing versions.
10. If a different Python version is needed, explain the reason without deleting the existing environment.
11. Explain hosted-model costs before any paid API calls. Never request real API keys in chat or display their contents.
12. Do not commit or push unless requested.

## Building order

- Verify the environment and existing files.
- Normalize the requirements filename if necessary.
- Create a fictional hospital document.
- Build document loading.
- Build chunking and retain source metadata.
- Configure embeddings.
- Build and persist the Chroma index.
- Test retrieval before connecting answer generation.
- Connect the language model.
- Generate grounded answers with sources and handle missing information.
- Add a terminal interface.
- Add sample questions and evaluation.
- Introduce LangGraph state, nodes, edges, and tool routing.
- Finish documentation and guide GitHub publishing when requested.

Build the document index separately and reuse it for questions. Do not re-embed the entire document collection for each question.

Conceptual reference: `https://medium.com/@wendell_89912/building-an-agentic-rag-with-langgraph-a-step-by-step-guide-009c5f0cce0a`

## Response format for each step

- Objective
- File being created or changed
- Why it is needed, with a simple example
- Code explanation
- Command to run
- Expected result
- What the user should send back
