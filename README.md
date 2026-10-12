# Hospital Information RAG

This beginner project answers questions from a fictional hospital document instead of relying only on an AI model's general knowledge. It builds a local searchable Chroma index, retrieves relevant passages, and asks an OpenAI model to write a short answer with citations. A LangGraph workflow also remembers the current terminal conversation and decides when a hospital search is needed.

> **Demo only:** Evergreen Community Medical Center, its portal, staff, phone numbers, prices, and policies are fictional. The application cannot check live availability, book appointments, or provide medical advice.

## Example

**Question**

```text
Where is the cardiology department?
```

**Example answer**

```text
The cardiology department is in the East Tower, 3rd Floor. [1]
```

The terminal then prints a reference key showing which current-response passage `[1]` refers to. Wording and retrieved passage order can vary between runs.

## What is implemented

- Loads the fictional hospital document and retains its source filename.
- Splits the document at Markdown headings and keeps section metadata.
- Uses OpenAI `text-embedding-3-small` embeddings.
- Persists a Chroma collection named `hospital_info` in `rag-pipeline/chroma_db/`.
- Reuses unchanged indexed chunks and removes outdated chunks for the source document.
- Retrieves up to six passages with semantic similarity search.
- Uses an OpenAI chat model to decide whether to search and to generate answers.
- Uses LangGraph tool calling for the hospital search workflow.
- Keeps follow-up context in process with `InMemorySaver`.
- Shows current-response citation labels and a source reference key.
- Supports `new` for a fresh conversation ID and `exit` or `quit` to stop.
- Runs interactive evaluation cases and incrementally saves JSON reports.
- Sends traces to LangSmith when optional tracing configuration is enabled.

## Learn the flow

- [Application flowchart](rag-pipeline/docs/flowchart.md)
- [Standalone architecture diagram](rag-pipeline/docs/architecture.mmd)
- [Rendered architecture SVG](rag-pipeline/docs/architecture.svg)

## Small glossary

| Term | Simple meaning |
|---|---|
| RAG | Retrieval-Augmented Generation: find relevant source text first, then ask an LLM to answer from it. |
| Chunk | A smaller passage cut from a larger document so it can be searched. |
| Embedding | A list of numbers representing the meaning of text, used for similarity matching. |
| Vector database | A database that stores embeddings and finds nearby, meaning-related text; this project uses Chroma. |
| LLM | Large language model: the model that decides whether to search and writes the final response. |
| Tool | A Python function the model can choose to call; here it is `search_hospital`. |
| LangChain | Libraries used for documents, OpenAI integrations, tools, and Chroma integration. |
| LangGraph | The library that connects stateful steps and routes the agent between answering and searching. |
| Trace | An observability record of a run's graph steps, tool calls, and model calls, sent to LangSmith when enabled. |

## Project tree

Generated cache contents are shortened, but every project area is shown.

```text
.
├── .git/                        # Git metadata
├── .gitignore
├── README.md
├── hospital-ui/                 # Browser UI and backend adapter
└── rag-pipeline/                # Complete RAG implementation and setup
    ├── .env                     # Private local secrets; Git-ignored
    ├── .env.example             # Safe configuration template
    ├── .venv/                   # Local Python environment; Git-ignored
    ├── requirements.txt
    ├── app/                     # Terminal chat entry point
    ├── chroma_db/               # Generated persistent vector index
    ├── config/                  # Non-secret configuration
    ├── data/                    # Source data and evaluation artifacts
    ├── docs/                    # Pipeline documentation and diagrams
    ├── scripts/                 # Index builder
    └── src/                     # RAG implementation packages
```

Python `__pycache__/` directories and `.pyc` files may also appear after running code; Git ignores them. Each source package also contains an `__init__.py` file so Python can import it consistently.

## Folder guide

| Folder | Purpose and when it is used |
|---|---|
| `.git/` | Git's internal version-control data; Git manages it automatically. |
| `hospital-ui/` | Browser interface and adapter; intentionally kept separate from the pipeline. |
| `rag-pipeline/` | Self-contained RAG code, data, index, settings, scripts, and documentation. |
| `rag-pipeline/.venv/` | Local Python environment containing the pipeline dependencies. |
| `rag-pipeline/app/` | User-facing terminal application; run it to chat. |
| `rag-pipeline/chroma_db/` | Generated local Chroma index; built separately and queried during chat. |
| `rag-pipeline/config/` | Reserved for non-secret settings. |
| `rag-pipeline/data/` | Source data, evaluation inputs, and generated evaluation output. |
| `rag-pipeline/docs/` | Pipeline setup notes, requirements history, and architecture documentation. |
| `rag-pipeline/scripts/` | Pipeline tasks, currently the separate index builder. |
| `rag-pipeline/src/` | Ingestion, retrieval, generation, evaluation, and LangGraph code. |
| `__pycache__/` | Generated Python bytecode caches; safe to ignore and excluded from Git. |

## Important Python files

| File | What it does |
|---|---|
| `app/main.py` | Loads `.env`, starts the memory-enabled graph, accepts terminal input, handles `new`, and prints answers and source keys. |
| `scripts/build_index.py` | Loads and chunks the hospital document, embeds only changed chunks, writes Chroma records, and deletes obsolete source records. |
| `src/agents/agent_graph.py` | Implements the prepare → agent → optional search → agent flow and the `search_hospital` tool. |
| `src/embeddings/embedding_model.py` | Validates `OPENAI_API_KEY` and creates `text-embedding-3-small`. |
| `src/evaluation/evaluate.py` | Runs single- and multi-turn cases, checks expected search behavior, asks for a human verdict, and saves progress to JSON. |
| `src/generation/llm_client.py` | Validates `OPENAI_API_KEY` and creates the configured OpenAI chat model. |
| `src/generation/rag_chain.py` | Provides a simpler standalone retrieval-and-generation function; the terminal app currently uses `agent_graph.py` instead. |
| `src/ingestion/document_loader.py` | Reads a UTF-8 text file and returns its text plus filename metadata. |
| `src/processing/chunking.py` | Creates chunks at Markdown headings and records section and subsection names. |
| `src/retrieval/retriever.py` | Queries Chroma with semantic similarity and returns the closest documents. |
| `src/vector_store/vector_db.py` | Connects the embedding model to the persisted Chroma collection. |

## Setup from a clean clone

Run all pipeline commands from `rag-pipeline/`.

### 1. Clone the repository

```bash
git clone https://github.com/dilipvavi8-dot/RAG-Basic-project-Hospital-.git
cd RAG-Basic-project-Hospital-
cd rag-pipeline
```

### 2. Create and activate a Python environment

The current workspace uses Python 3.14.3.

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

### 3. Install the declared dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The direct dependencies currently declared by the project are:

| Dependency | Declared version |
|---|---|
| `langchain-openai` | `1.7.0` |
| `langchain-chroma` | `1.1.0` |
| `langgraph` | `1.2.14` |
| `python-dotenv` | Unpinned |

Packages such as `langchain-core`, Chroma's underlying dependencies, and `langsmith` are installed transitively.

### 4. Create the private environment file

```bash
cp .env.example .env
```

Use placeholder structure like this, then put your real values only in your local `.env`:

```dotenv
OPENAI_API_KEY=your-openai-api-key

# Optional LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=hospital-rag-learning
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# LANGSMITH_WORKSPACE_ID=your-workspace-id
```

Use `https://eu.api.smith.langchain.com` instead when your LangSmith account is in the EU region. A workspace ID is needed only when the key's scope requires choosing a workspace.

Never commit `.env`, paste its contents into issues, or print its keys in logs. The repository's `.gitignore` excludes `rag-pipeline/.env`, `.venv/`, `rag-pipeline/chroma_db/`, caches, and macOS metadata while allowing `rag-pipeline/.env.example`.

### 5. Build the index separately

```bash
python -m scripts.build_index
```

Expected output includes the total chunk count, how many chunks need embeddings, how many outdated records were removed, and `Indexing complete!`. This command uses the OpenAI embedding API and may incur API charges. Run it before the first chat and again after changing the hospital document.

### 6. Start the terminal application

```bash
python -m app.main
```

Ask a hospital-information question. Type `new` for a fresh conversation ID, or `exit`/`quit` to stop. Model calls may incur API charges.

### 7. Run the interactive evaluation

```bash
python -m src.evaluation.evaluate
```

The evaluator runs the cases in `rag-pipeline/data/sample_queries/questions.json`, displays expected facts and prohibited claims, checks whether search behavior matched `expected_search`, and asks you to mark answer quality as `pass`, `fail`, or `unsure`. It saves progress after every reviewed question under `rag-pipeline/data/evaluation_results/`. Evaluation uses model and embedding calls and may incur API charges.

### Optional: inspect LangSmith traces

When valid `LANGSMITH_*` values are present, LangChain and LangGraph send observability traces to the configured LangSmith project. LangSmith helps inspect the workflow; it does not retrieve hospital passages or generate hospital answers. Disable tracing locally by removing those optional variables or setting `LANGSMITH_TRACING=false`.

## How conversation and retrieval state differ

- `rag-pipeline/chroma_db/` is persistent local storage. It survives application restarts and contains indexed hospital chunks and embeddings.
- Conversation messages use LangGraph's `InMemorySaver`. They are available only while the current Python process runs.
- `new` generates another UUID and begins using a fresh conversation thread. Previous messages are not used by that new thread.
- When the process exits, all in-memory conversation state disappears. This project does not implement persistent chat history.

## Evaluation findings currently in the workspace

One saved report exists and is marked `running`, so it is an incomplete evaluation rather than a finished benchmark:

- 15 questions were planned and 12 had reviews saved.
- Human answer review: 11 pass, 1 fail, 0 unsure.
- Search-routing checks: 12 checked, with 2 mismatches.
- Both routing mismatches involved ambiguous “its Saturday hours” questions where a search occurred although the expected behavior was to ask for clarification without searching.
- One of those mismatched searches still received a human `pass`, demonstrating why answer correctness and search behavior are tracked separately.

These results do not prove perfect accuracy. The remaining questions were not recorded as reviewed, and model behavior can vary between runs.

## Known limitations

- The knowledge base contains one fictional hospital document, not real clinical or operational data.
- The demo portal and all contact details are fictional; there is no connection to a live scheduling system.
- The app cannot check appointment availability or book, cancel, or modify appointments.
- The chat model can still make a routing mistake, such as searching when it should ask for clarification.
- Retrieval uses one semantic similarity search only—there is no hybrid search, reranker, or query fusion.
- Chunking follows headings rather than enforcing a token limit or overlap.
- Conversation memory is process-local and is lost on exit.
- Model, embedding, chunking, and retrieval settings are mostly hardcoded; `config/settings.yaml` is currently unused.
- OpenAI access is required for indexing, querying, generation, and most evaluation cases.
- The evaluator uses human verdicts; it does not automatically prove factual correctness.

## Possible next steps

1. Complete and review a full evaluation run before changing retrieval behavior.
2. Improve the prompt or routing checks for ambiguous follow-up questions.
3. Move non-secret hardcoded settings into `config/settings.yaml`.
4. Add automated checks for citations, expected facts, and forbidden claims.
5. Add token-aware chunk sizing only if evaluation shows heading chunks are too large or too small.
6. Consider persistent chat history only if the learning project needs conversations across restarts.

These are proposals, not currently implemented features.

## Reusable checklist for another document collection

- [ ] Define the questions the assistant should and should not answer.
- [ ] Add trustworthy source documents and record source metadata.
- [ ] Choose chunk boundaries that preserve useful context.
- [ ] Select an embedding model and vector database.
- [ ] Build the index separately from the chat loop.
- [ ] Reuse unchanged vectors instead of embedding everything on every question.
- [ ] Test retrieval before relying on generated answers.
- [ ] Give the model only retrieved evidence and clear missing-information rules.
- [ ] Display citations that map to the passages used for the current answer.
- [ ] Decide whether conversation history should be temporary or persistent.
- [ ] Add test questions for facts, ambiguity, follow-ups, missing information, and unsafe requests.
- [ ] Evaluate answer correctness separately from search and routing behavior.
- [ ] Keep secrets in `.env`, publish only placeholders, and verify `.gitignore`.
- [ ] Add tracing for debugging, while remembering that tracing is observability rather than knowledge retrieval.
