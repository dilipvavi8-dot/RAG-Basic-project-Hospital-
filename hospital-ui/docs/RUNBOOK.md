# Run the hospital UI locally

## 1. Open the right folder

The `hospital-ui` folder is now inside `RAG-Basic-project-Hospital-`. All commands below run from `hospital-ui` unless stated otherwise. This remains a separate local Git repository; no remote is required.

```bash
cd /path/to/hospital-testing/RAG-Basic-project-Hospital-/hospital-ui
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r ../rag-pipeline/requirements.txt
```

The original workspace uses Python 3.14.3. Install both dependency files into this UI environment so the imported graph has all its libraries. Do not copy the original environment or Chroma files.

## 2. Configure the connection

For a first setup only:

```bash
cp .env.example .env
```

Edit `.env` in your editor:

```dotenv
RAG_PROJECT_PATH=../rag-pipeline
CHAT_TIMEOUT_SECONDS=120
```

An absolute path ending in `/RAG-Basic-project-Hospital-/rag-pipeline` also works. Relative paths resolve from `hospital-ui`, regardless of where the server command was launched.

The original RAG `.env` supplies `OPENAI_API_KEY` and optional `LANGSMITH_*` settings. Any backend override belongs in the UI `.env`, never in JavaScript or HTML. Do not share keys in chat, screenshots, Git, or error reports. Verify exclusion with `git check-ignore .env`.

## 3. Check the original index

If the original app already works and has a populated `hospital_info` collection, skip building. The adapter uses the original `chroma_db/chroma.sqlite3` in place and checks that the collection has records.

If the index is missing, run the existing builder from the original repository using the UI environment:

```bash
cd ../rag-pipeline
../hospital-ui/.venv/bin/python -m scripts.build_index
cd ../hospital-ui
```

This step calls the OpenAI embedding API and can incur charges. Run indexing separately from chatting and after intentional document changes. The UI never automatically rebuilds your index.

## 4. Start the server

```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open **http://127.0.0.1:8000**. The top status bar should say it is connected to the hospital RAG pipeline. Open **http://127.0.0.1:8000/guide/** for the readable flow diagram. Stop the server with Ctrl+C.

Use exactly one worker because the graph memory, conversation locks, and retry cache are process-local. Avoid `--reload` while testing conversation memory: reloads restart the process. Do not bind to `0.0.0.0` or publish this server; it has no account authentication.

## 5. Run free automated checks

```bash
python -m pytest -q
```

These use a mocked graph and no paid model calls. They cover response fields, citation marking, validation, conversation isolation, duplicate request reuse, serialization, timeouts, sanitized failures, and startup errors.

## 6. Manual checks with the real pipeline

These checks can incur OpenAI API charges. Expected behavior is a test target, not a guarantee.

| Check | Action and expected result |
| --- | --- |
| Greeting | Send `Hello`; expect a brief greeting, usually without search. |
| Cited hospital answer | Ask `Where is the cardiology department?`; expect East Tower, 3rd Floor and a supporting reference. |
| Missing information | Ask `What is Dr. Smith's phone number?`; expect missing-information wording, not an invented number. |
| Follow-up memory | Ask about dermatology, then `What are its Saturday hours?`; expect the follow-up to refer to dermatology. |
| Reset | Click New Conversation, then ask `What are its Saturday hours?`; expect clarification without reusing the prior topic. |
| Disconnected backend | Stop the server, send a message, and check the error. Restart and click New Conversation; old server IDs are no longer valid. |
| Retry without duplication | Temporarily interrupt the connection during a request, restore it while the same server runs, and click Retry. One user bubble remains and the same request ID is reused. |
| Keyboard | Enter sends; Shift+Enter adds a newline; Tab reaches buttons, details, and the input with visible focus. |
| References | Open references and compare “Cited in answer” with “Retrieved only”; reference labels apply only to that response. |

Use `python -m src.evaluation.evaluate` from `rag-pipeline/` for repeatable, saved human evaluation reports. The UI's response details and exploratory chats are not an evaluation report.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Missing project path | Set `RAG_PROJECT_PATH` to the folder containing `src/agents/agent_graph.py`; restart. |
| Missing dependencies | Activate the UI environment and install both requirements files. |
| Empty or missing index | Run the original index builder; check the original collection name is `hospital_info`. |
| Chroma inspection fails | Confirm the original index works and both environments use compatible original dependency versions; do not move the database. |
| Missing key or authentication failure | Use exact uppercase `OPENAI_API_KEY` / `LANGSMITH_API_KEY`; save `.env`, then restart. Existing shell variables override `.env`, even if empty. |
| OpenAI connection fails | Check internet access, your account's model access, API billing/quota, and any configured proxy/base URL privately. Browser errors intentionally omit exception details. |
| Ollama is running but no answers | The current RAG code uses OpenAI. Ollama is not configured by this adapter. |
| LangSmith errors | Check the key and correct US/EU endpoint in backend configuration, or set `LANGSMITH_TRACING=false` in UI `.env` while debugging. |
| Timeout | Click Retry to wait for the same request. Work may continue after the HTTP timeout and retain the conversation lock. |
| Model/retrieval failure | Fix connectivity/credentials and start New Conversation; partial graph state is not silently replayed. |
| Expired conversation | Server restarts clear IDs. Click New Conversation. |
| Port occupied | Use `--port 8001`, then open `http://127.0.0.1:8001`. |
| Diagram shows code in editor | Open `docs/flowchart.svg` in a browser or visit `/guide/`; the SVG needs no Mermaid extension. |

## API contract

`POST /api/conversations` returns a new `conversation_id`. `POST /api/chat` accepts `{ "question": "Hello", "conversation_id": "UUID", "request_id": "UUID" }`. A successful response has `answer`, `sources`, `search_used`, `elapsed_seconds`, and `conversation_id`. Each source includes `label`, `filename`, `section`, `subsection`, and `cited`. Reuse the exact request ID and question for a retry. Questions must contain 1–2000 characters. `GET /api/health` returns a readiness boolean and actionable setup message.

The HTTP timeout bounds waiting, not worker-thread lifetime. Model-side timeouts and retries belong to the existing RAG client. Restarting can abandon work and clears memory. No durable jobs or public multi-user authentication are implemented.
