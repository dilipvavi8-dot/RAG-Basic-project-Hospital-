# Complete authored file contents

> Historical creation snapshot: the UI has since moved inside the RAG project. Use the current README, RUNBOOK.md, `.env.example`, and source files for the updated nested paths. Current validation results are in the parent project's `docs/VALIDATION_2026-10-12.md`.

Full files, not diffs. Private `.env`, installed environments, Git internals, and generated verification artifacts are excluded. See `.env.example` for safe configuration. This compilation does not include itself.

## .gitignore

`````
.env
.env.*
!.env.example
.venv/
__pycache__/
*.pyc
.pytest_cache/
.playwright-cli/
output/
.DS_Store

`````

## .env.example

`````
RAG_PROJECT_PATH=../RAG-Basic-project-Hospital-
CHAT_TIMEOUT_SECONDS=120
# Credentials are read from the existing RAG project's .env.
# Optional backend-only overrides; never put secrets in frontend files.
# OPENAI_API_KEY=your-openai-key
# LANGSMITH_API_KEY=your-langsmith-key

`````

## requirements.txt

`````
fastapi>=0.115,<1
uvicorn>=0.30,<1
python-dotenv>=1,<2
httpx>=0.27,<1
pytest>=8,<10

`````

## README.md

`````
# Hospital Information Assistant — local web UI

A browser interface for the existing fictional hospital RAG project. It uses the original LangGraph agent, OpenAI models, and original Chroma database. The UI does not invent hospital answers or contain a second knowledge base.

![Simple flow diagram](docs/flowchart.svg)

Open [the diagram](docs/flowchart.svg) directly if your Markdown viewer does not show images. Once the server runs, [the visual guide](http://127.0.0.1:8000/guide/) displays the diagram as an ordinary image: no Mermaid plugin is required. [Mermaid source](docs/flowchart.mmd) is also available.

## Start here

Follow the complete [local runbook](docs/RUNBOOK.md). The folders should be siblings:

```text
hospital-testing/
├── RAG-Basic-project-Hospital-/    existing project, data, .env, chroma_db
└── hospital-ui/                   this independent Git repository
```

From `hospital-ui`, on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r ../RAG-Basic-project-Hospital-/requirements.txt
cp .env.example .env
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Only copy `.env.example` for a new setup; do not overwrite an existing `.env`. Edit `RAG_PROJECT_PATH` if your original repository has a different name or location. Open **http://127.0.0.1:8000**. No Node, React, build step, or paid UI service is required.

Configuration example (placeholders only):

```dotenv
RAG_PROJECT_PATH=../RAG-Basic-project-Hospital-
CHAT_TIMEOUT_SECONDS=120
# Optional overrides; otherwise reuse the original project's .env:
# OPENAI_API_KEY=your-openai-key
# LANGSMITH_API_KEY=your-langsmith-key
```

Environment precedence is the shell, then `hospital-ui/.env`, then the original RAG `.env`. Existing variables, including empty ones, take precedence. Keys must use their exact uppercase spelling. Credentials are loaded before importing the original graph and never sent to the frontend.

## Project folders

| Folder | Purpose |
| --- | --- |
| `backend/` | FastAPI routes and an adapter that imports the existing graph. |
| `frontend/` | Plain HTML, CSS, and JavaScript served by the same Python server. |
| `tests/` | Mocked backend tests with no model or embedding calls. |
| `docs/` | Local runbook and diagrams visible without Mermaid support. |
| `.venv/` | Ignored Python environment for both projects' dependencies. |
| `output/` | Ignored verification artifacts, when generated. |

`backend/main.py` handles validated chat requests, conversation IDs, safe errors, request reuse, and timeouts. `backend/rag_adapter.py` checks configuration and index availability, then calls `build_graph(enable_memory=True)` from the existing project.

## Behavior

- Each conversation receives its own UUID used as a LangGraph thread ID. Requests within that conversation are serialized; graph calls run in worker threads rather than blocking the async event loop.
- A request UUID prevents duplicate graph execution when a timed-out or disconnected request is retried. Retry rechecks the same job; it does not append another user bubble.
- The HTTP wait defaults to 120 seconds. A timeout does not cancel a running model call or release its conversation lock. Existing model timeouts and retries still apply. A failed graph can have partial state, so start a New Conversation after a model or retrieval failure.
- New Conversation selects a new thread and clears visible messages. Old threads remain in server memory until shutdown. A server restart loses all chats, request caches, and IDs. Refreshing the browser also starts a fresh visible session.
- There are local limits of 200 conversations and 100 requests per conversation to bound request cache growth; restart the server to clear it.
- The existing graph makes at most one search per question. It uses history to resolve follow-ups, but final generation uses current-turn passages. Greetings, clarification, and scope replies may bypass retrieval.
- All reference entries were retrieved; “Cited in answer” means that numeric label appears in the answer. This detects citation labels, not evidence quality. Citation numbers belong only to the current response.
- Output uses safe text nodes, preserves paragraphs and line breaks, and shows literal Markdown punctuation when present. Fictional `.example` addresses remain plain text.

## Testing and limitations

```bash
python -m pytest -q
```

Tests use a mocked graph only. Production has no simulated-answer fallback. See the runbook for manual checks of real answers, missing facts, follow-ups, reset behavior, and disconnection.

This is a local development app with no login, public deployment support, or persistent conversation database. Use one server worker and bind to localhost. The hospital, portal, doctors, and contact information are fictional. There is no live availability or booking tool. Errors and model decisions can still occur, including an unnecessary search for ambiguous questions.

OpenAI model and embedding calls can cost money; even a greeting can call the model. LangSmith tracing is optional and may transmit questions, passages, and answers to that external service. The browser UI helps manual exploration but does not replace the existing saved evaluations (`python -m src.evaluation.evaluate` in the RAG repository).

The inspected pipeline uses OpenAI (`gpt-6-luna` and `text-embedding-3-small`), not Ollama. Starting Ollama alone will not satisfy its API requirements. This UI does not change providers.

Implementation references: [FastAPI static files](https://fastapi.tiangolo.com/tutorial/static-files/) and [application lifespan](https://fastapi.tiangolo.com/advanced/events/).

See [verification notes](docs/VERIFICATION.md) for what was actually checked. [Complete file contents](docs/COMPLETE_SOURCE.md) provides the full authored source files together, excluding private `.env`, generated environments, and browser artifacts.

`````

## backend/__init__.py

`````
"""Local web adapter for the existing hospital RAG application."""

`````

## backend/main.py

`````
"""Same-origin local UI and API. Run with one Uvicorn worker."""
import asyncio
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.rag_adapter import ROOT, SetupError, load_graph


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: UUID
    request_id: UUID

    @field_validator('question')
    @classmethod
    def trim_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('Question cannot be blank')
        return value


@dataclass
class Conversation:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    requests: dict = field(default_factory=dict)
    failed: bool = False


def create_app(graph=None, timeout=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.graph = graph
        app.state.setup_error = None
        app.state.conversations = {}
        if graph is None:
            try:
                app.state.graph = await asyncio.to_thread(load_graph)
            except SetupError as exc:
                app.state.setup_error = str(exc)
        try:
            app.state.timeout = float(timeout if timeout is not None else os.getenv('CHAT_TIMEOUT_SECONDS', '120'))
            if not 0 < app.state.timeout <= 600:
                raise ValueError
        except ValueError:
            app.state.setup_error = 'CHAT_TIMEOUT_SECONDS must be greater than zero and at most 600.'
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        origin = request.headers.get('origin')
        if request.method == 'POST' and origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': 'Open this app from its local server address.'}, status_code=403)
        response = await call_next(request)
        response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'"
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({'detail': 'Use a question of 1–2000 characters and valid conversation and request IDs.'}, status_code=422)

    @app.get('/api/health')
    async def health():
        return {'ready': app.state.graph is not None and app.state.setup_error is None,
                'message': app.state.setup_error or 'Connected to the hospital RAG pipeline.'}

    @app.post('/api/conversations', status_code=201)
    async def new_conversation():
        if app.state.setup_error:
            raise HTTPException(503, app.state.setup_error)
        if len(app.state.conversations) >= 200:
            raise HTTPException(429, 'Local conversation limit reached. Restart the server to clear memory.')
        identifier = str(uuid4())
        app.state.conversations[identifier] = Conversation()
        return {'conversation_id': identifier}

    async def run_turn(conversation, payload):
        async with conversation.lock:
            if conversation.failed:
                return {'error': 'This conversation stopped after an error. Start a New Conversation.'}
            started = time.monotonic()
            try:
                result = await asyncio.to_thread(
                    app.state.graph.invoke, {'question': payload.question},
                    config={'configurable': {'thread_id': str(payload.conversation_id)}},
                )
                answer = result['answer']
                if not isinstance(answer, str):
                    raise ValueError('Unexpected answer shape')
                labels = set(re.findall(r'\[(\d+)\]', answer))
                sources = [{**{key: source.get(key, '') for key in ('label', 'filename', 'section', 'subsection')},
                            'cited': str(source.get('label')) in labels}
                           for source in result.get('sources', [])]
                return {'answer': answer, 'sources': sources,
                        'search_used': bool(result.get('searched', False)),
                        'elapsed_seconds': round(time.monotonic() - started, 2),
                        'conversation_id': str(payload.conversation_id)}
            except Exception:
                # A graph error can leave partial state. Never silently replay it.
                conversation.failed = True
                return {'error': 'The model or retrieval service could not complete this request. Check backend credentials and connectivity, then start a New Conversation.'}

    @app.post('/api/chat')
    async def chat(payload: ChatRequest):
        if app.state.setup_error:
            raise HTTPException(503, app.state.setup_error)
        conversation = app.state.conversations.get(str(payload.conversation_id))
        if conversation is None:
            raise HTTPException(404, 'Conversation expired or does not exist. Start a New Conversation.')
        request_id = str(payload.request_id)
        saved = conversation.requests.get(request_id)
        if saved:
            question, task = saved
            if question != payload.question:
                raise HTTPException(409, 'A request ID cannot be reused for a different question.')
        else:
            if len(conversation.requests) >= 100:
                raise HTTPException(429, 'Conversation limit reached. Start a New Conversation.')
            task = asyncio.create_task(run_turn(conversation, payload))
            conversation.requests[request_id] = (payload.question, task)
        try:
            result = await asyncio.wait_for(asyncio.shield(task), app.state.timeout)
        except TimeoutError:
            raise HTTPException(504, 'Still waiting for the model. Retry to check the same request; it will not be sent twice.') from None
        if 'error' in result:
            raise HTTPException(502, result['error'])
        return result

    app.mount('/guide', StaticFiles(directory=ROOT / 'docs', html=True), name='guide')
    app.mount('/', StaticFiles(directory=ROOT / 'frontend', html=True), name='frontend')
    return app


app = create_app()

`````

## backend/rag_adapter.py

`````
"""Import the real graph only after loading backend configuration."""
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


class SetupError(Exception):
    """An actionable configuration error safe to show locally."""


def load_graph():
    load_dotenv(ROOT / '.env')
    configured = os.getenv('RAG_PROJECT_PATH', '').strip()
    if not configured:
        raise SetupError('Set RAG_PROJECT_PATH in hospital-ui/.env, then restart.')
    project = Path(configured).expanduser()
    if not project.is_absolute():
        project = ROOT / project
    project = project.resolve()
    if not (project / 'src/agents/agent_graph.py').is_file():
        raise SetupError('RAG_PROJECT_PATH must point to the existing RAG repository root.')
    load_dotenv(project / '.env')
    if not os.getenv('OPENAI_API_KEY', '').strip():
        raise SetupError('OPENAI_API_KEY is missing or empty. Check backend environment configuration and restart.')
    if os.getenv('LANGSMITH_TRACING', '').lower() == 'true' and not os.getenv('LANGSMITH_API_KEY', '').strip():
        raise SetupError('LangSmith tracing is enabled without a key. Set LANGSMITH_API_KEY or disable tracing, then restart.')
    database = project / 'chroma_db/chroma.sqlite3'
    if not database.is_file():
        raise SetupError('Hospital index is missing. Run python -m scripts.build_index in the RAG repository first.')
    # Read-only check avoids creating or modifying a collection at startup.
    try:
        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
            count = connection.execute(
                'SELECT COUNT(*) FROM embeddings e JOIN segments s ON e.segment_id=s.id '
                'JOIN collections c ON s.collection=c.id WHERE c.name=?',
                ('hospital_info',),
            ).fetchone()[0]
        if not count:
            raise SetupError('The hospital_info index is empty. Run the RAG index builder first.')
    except sqlite3.Error:
        raise SetupError('Cannot read the Chroma index. Check the original RAG index and dependency versions.') from None
    sys.path.insert(0, str(project))
    sys.dont_write_bytecode = True
    try:
        from src.agents.agent_graph import build_graph
        return build_graph(enable_memory=True)
    except ImportError:
        raise SetupError('RAG dependencies are missing. Install both requirements files in the UI virtual environment.') from None
    except Exception:
        raise SetupError('RAG initialization failed. Check backend credentials, model configuration, and installed dependencies.') from None

`````

## frontend/app.js

`````
'use strict';
const $ = (id) => document.getElementById(id);
let conversationId = null;
let busy = false;
let pending = null;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function setBusy(value) {
  busy = value;
  $('send').disabled = value;
  $('new-chat').disabled = value;
  $('retry').disabled = value;
  document.querySelectorAll('[data-question]').forEach(button => button.disabled = value);
  $('chat-form').setAttribute('aria-busy', String(value));
}
async function api(path, body) {
  let response;
  try {
    response = await fetch(path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: {'Content-Type': 'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(610000),
    });
  } catch {
    throw new Error('Cannot reach the local backend. Check that the server is running, then retry.');
  }
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed. Please try again.');
  return data;
}
function error(message) {
  $('error-text').textContent = message;
  $('error').hidden = false;
  $('retry').hidden = !pending;
}
function bubble(role, text) {
  $('welcome').hidden = true;
  const item = node('article', undefined, `message ${role}`);
  item.append(node('div', role === 'user' ? 'You' : 'Hospital assistant', 'speaker'));
  item.append(node('div', text, 'text'));
  $('messages').append(item);
  item.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  return item;
}
function showAnswer(data) {
  const item = bubble('assistant', data.answer);
  if (data.sources.length) {
    const references = node('details');
    const cited = data.sources.filter(source => source.cited).length;
    references.append(node('summary', `References · ${cited} cited / ${data.sources.length} retrieved`));
    references.append(node('p', 'Numbers apply only to this answer. “Cited” means its label appears in the answer; it does not prove the claim is correct.', 'citation-note'));
    data.sources.forEach(source => {
      const row = node('div', undefined, 'source');
      row.append(node('strong', `[${source.label}] ${source.filename} · ${source.cited ? 'Cited in answer' : 'Retrieved only'}`));
      row.append(node('div', [source.section, source.subsection].filter(Boolean).join(' / ') || 'No section metadata'));
      references.append(row);
    });
    item.append(references);
  }
  const details = node('details');
  details.append(node('summary', 'Response details'));
  details.append(node('p', `Hospital search: ${data.search_used ? 'used' : 'not used'} · Response time: ${data.elapsed_seconds.toFixed(2)} seconds`));
  item.append(details);
}
async function send(retry = false) {
  if (busy) return;
  if (!retry) {
    const question = $('question').value.trim();
    if (!question) return;
    if (pending) { error('Retry the pending question or start a New Conversation before sending another.'); return; }
    pending = {question, request_id: crypto.randomUUID()};
    bubble('user', question);
    $('question').value = '';
  }
  if (!pending) return;
  setBusy(true);
  $('error').hidden = true;
  const loading = node('div', 'Waiting for the hospital assistant…', 'pending');
  loading.setAttribute('role', 'status');
  $('messages').append(loading);
  loading.scrollIntoView({block: 'nearest'});
  try {
    if (!conversationId) conversationId = (await api('/api/conversations', {})).conversation_id;
    const data = await api('/api/chat', {...pending, conversation_id: conversationId});
    showAnswer(data);
    pending = null;
  } catch (exc) { error(exc.message); }
  finally { loading.remove(); setBusy(false); $('question').focus(); }
}
async function newConversation() {
  if (busy) return;
  setBusy(true);
  try {
    conversationId = (await api('/api/conversations', {})).conversation_id;
    pending = null;
    $('messages').replaceChildren();
    $('welcome').hidden = false;
    $('error').hidden = true;
    $('question').value = '';
    $('connection').textContent = 'New conversation ready. Previous messages will not be used.';
    $('connection').classList.remove('problem');
  } catch (exc) { error(exc.message); }
  finally { setBusy(false); $('question').focus(); }
}
$('chat-form').addEventListener('submit', event => {event.preventDefault(); send();});
$('question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault(); send();}
});
$('new-chat').addEventListener('click', newConversation);
$('retry').addEventListener('click', () => send(true));
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => {
  $('question').value = button.dataset.question;
  $('question').focus();
}));
api('/api/health').then(data => {
  $('connection').textContent = data.message;
  $('connection').classList.toggle('problem', !data.ready);
}).catch(exc => {
  $('connection').textContent = exc.message;
  $('connection').classList.add('problem');
});

`````

## frontend/icon.svg

`````
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40"><rect width="40" height="40" rx="10" fill="#24594b"/><path d="M20 10v20M10 20h20" stroke="white" stroke-width="4"/></svg>

`````

## frontend/index.html

`````
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hospital Information Assistant</title>
  <link rel="icon" href="/icon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/styles.css">
  <script src="/app.js" defer></script>
</head>
<body>
  <aside class="sidebar">
    <a class="brand" href="/" aria-label="Hospital assistant home"><span class="mark">+</span><span>Evergreen<span class="brand-sub">COMMUNITY MEDICAL CENTER</span></span></a>
    <span class="demo-tag">FICTIONAL HOSPITAL · LEARNING DEMO</span>
    <div class="side-section"><span class="eyebrow">YOUR WORKSPACE</span><div class="active-nav"><span aria-hidden="true">◌</span> Information assistant</div></div>
    <button id="new-chat" class="new-button" type="button">＋ New Conversation</button>
    <div class="side-note"><span class="eyebrow">A LITTLE CONTEXT</span><p>Ask about departments, visiting hours, and appointment instructions.</p><p>Follow-up questions work within this conversation.</p></div>
    <a class="guide-link" href="/guide/" target="_blank" rel="noopener">How the assistant works <span aria-hidden="true">↗</span></a>
    <div class="side-footer"><span class="status-dot"></span> Local learning workspace<p>Conversation memory clears when the server restarts.</p></div>
  </aside>
  <main>
    <header><div><span class="eyebrow">HOSPITAL GUIDE</span><h1>Hospital Information Assistant</h1></div><span class="private-label">Local demo</span></header>
    <div id="connection" class="connection" role="status">Connecting to your local backend…</div>
    <section class="chat-area" aria-label="Conversation">
      <div id="welcome" class="welcome">
        <div class="welcome-icon" aria-hidden="true">✳</div>
        <span class="eyebrow">A HELPFUL PLACE TO START</span>
        <h2>Hospital information,<br><em>one question at a time.</em></h2>
        <p>Find your way through our fictional hospital.<br>Answers come with references you can check.</p>
        <div class="examples" aria-label="Example questions">
          <button type="button" data-question="Where is the cardiology department?"><span class="card-label">FIND A DEPARTMENT <span>↗</span></span>Where is the cardiology department?</button>
          <button type="button" data-question="What are dermatology’s Saturday hours?"><span class="card-label">CHECK CLINIC HOURS <span>↗</span></span>What are dermatology’s Saturday hours?</button>
          <button type="button" data-question="How do I request an appointment?"><span class="card-label">PLAN A VISIT <span>↗</span></span>How do I request an appointment?</button>
        </div>
        <p class="reference-hint">Grounded in your hospital document · References below each answer</p>
      </div>
      <div id="messages" role="log" aria-live="polite" aria-relevant="additions" aria-label="Chat messages"></div>
    </section>
    <div class="composer-wrap">
      <div id="error" class="error" role="alert" hidden><span id="error-text"></span><button id="retry" type="button">Retry</button></div>
      <form id="chat-form" class="composer">
        <label for="question">Your question</label>
        <textarea id="question" rows="2" maxlength="2000" placeholder="Ask about a department, hours, or hospital services…" required></textarea>
        <div class="composer-bottom"><span>Enter to send <span class="separator">·</span> Shift + Enter for a new line</span><button id="send" type="submit">Send <span aria-hidden="true">↑</span></button></div>
      </form>
      <p class="disclaimer">Fictional hospital demo. No medical advice, live availability, or appointment booking.</p>
    </div>
  </main>
</body>
</html>

`````

## frontend/styles.css

`````
:root{color-scheme:light;--ink:#203d37;--muted:#586d66;--green:#24594b;--line:#dce4df;--paper:#fafbf8;--soft:#eff4ed;font-family:Arial,Helvetica,sans-serif}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);display:flex;min-height:100vh}button,a,textarea{-webkit-tap-highlight-color:transparent}button{font:inherit;cursor:pointer}button:disabled{cursor:wait;opacity:.55}a{color:inherit}button:focus-visible,a:focus-visible,textarea:focus-visible,summary:focus-visible{outline:3px solid #ba771f;outline-offset:4px}.sidebar{width:282px;flex-shrink:0;background:#f0f4ee;border-right:1px solid var(--line);padding:34px 24px;position:sticky;top:0;height:100vh;display:flex;flex-direction:column}.brand{text-decoration:none;display:flex;align-items:center;gap:10px;font-family:Georgia,serif;font-size:26px}.mark{font-family:Arial,sans-serif;background:var(--green);color:white;border-radius:10px;font-size:31px;display:grid;place-items:center;width:42px;height:42px}.brand-sub{display:block;font:8px Arial,sans-serif;letter-spacing:1.2px;margin-top:5px}.demo-tag{font-size:9px;letter-spacing:.8px;background:#e4ebdf;border:1px solid #d1ddcc;border-radius:4px;padding:8px 5px;margin-top:28px;text-align:center}.eyebrow{font-size:10px;letter-spacing:1.8px;font-weight:700;color:var(--muted)}.side-section{margin-top:45px}.active-nav{margin:16px -5px 12px;padding:14px 10px;border-radius:7px;background:#e1eadf;font-size:13px;display:flex;gap:12px}.new-button{border:1px solid #b6c7bb;background:transparent;border-radius:7px;text-align:left;padding:13px;font-size:13px;color:var(--ink)}.side-note{margin-top:40px;font-size:13px;line-height:1.7;color:var(--muted)}.guide-link{font-size:13px;text-decoration:none;display:flex;justify-content:space-between;padding:16px 0;border-top:1px solid var(--line);margin-top:22px}.side-footer{margin-top:auto;font-size:11px;padding-top:28px;color:var(--muted);line-height:1.7}.side-footer p{font-size:11px;max-width:195px}.status-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#588254;margin-right:6px}main{flex:1;min-width:0;display:flex;flex-direction:column}header{padding:26px 40px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;background:#fff}h1{font-size:18px;font-weight:500;margin:8px 0 0}.private-label{font-size:11px;border:1px solid var(--line);border-radius:20px;padding:7px 12px}.connection{font-size:12px;color:var(--muted);padding:12px 40px;background:#f3f6f0}.connection.problem{color:#8f3025;background:#fff0e8}.chat-area{width:100%;max-width:960px;margin:0 auto;padding:30px 40px;flex:1}.welcome{text-align:center;padding:20px 0 5px}.welcome-icon{width:56px;height:56px;border:1px solid #d6e1ce;background:#eef4e7;border-radius:18px;display:grid;place-items:center;margin:0 auto 23px;color:var(--green);font-size:36px}.welcome h2{font:42px/1.18 Georgia,serif;letter-spacing:-1.4px;margin:17px 0}.welcome h2 em{font-weight:normal;color:#5c7862}.welcome>p{font-size:14px;line-height:1.8;color:var(--muted)}.examples{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:32px 0 18px}.examples button{text-align:left;color:var(--ink);background:#fff;border:1px solid var(--line);border-radius:9px;padding:19px 16px;line-height:1.6;font-size:13px;transition:border-color .2s,transform .2s}.examples button:hover{border-color:#7f9c86;transform:translateY(-2px)}.card-label{display:flex;justify-content:space-between;font-size:9px;letter-spacing:1px;font-weight:700;color:var(--muted);margin-bottom:14px}.reference-hint{font-size:11px!important}.composer-wrap{position:sticky;bottom:0;padding:12px 40px 20px;background:var(--paper);width:100%;max-width:960px;margin:auto}.composer{background:white;border:1px solid #becfc1;border-radius:12px;padding:17px 18px;box-shadow:0 4px 22px #244b3910}.composer label{font-size:11px;color:var(--muted)}textarea{display:block;resize:vertical;width:100%;min-height:66px;max-height:180px;border:none;background:transparent;padding:12px 0;font:14px/1.6 Arial,sans-serif;color:var(--ink)}textarea::placeholder{color:#718079}.composer-bottom{display:flex;justify-content:space-between;align-items:center;gap:10px}.composer-bottom>span{font-size:10px;color:var(--muted)}.separator{margin:0 7px}#send{background:var(--green);color:#fff;border:0;border-radius:7px;padding:10px 18px;font-size:12px;display:flex;gap:20px}.disclaimer{text-align:center;font-size:10px;color:var(--muted);margin:12px 0 0}.message{margin:18px 0;line-height:1.65;font-size:14px}.message.user{margin-left:15%;background:#eaf0e6;border-radius:14px 14px 3px 14px;padding:16px 20px}.message.assistant{margin-right:5%;background:white;border:1px solid var(--line);border-radius:3px 14px 14px 14px;padding:20px}.message .speaker{font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:var(--muted);margin-bottom:8px}.message .text{white-space:pre-wrap;overflow-wrap:anywhere}.message details{margin-top:16px;border-top:1px solid var(--line);padding-top:10px;font-size:12px;color:var(--muted)}summary{cursor:pointer}.source{margin:9px 0;padding:10px;background:#f5f7f2;border-radius:6px;overflow-wrap:anywhere}.source strong{display:block;color:var(--ink)}.citation-note{font-size:11px}.pending{color:var(--muted);font-size:13px;padding:14px 0}.error{background:#fff1e9;border:1px solid #e8c9b8;color:#813b27;border-radius:7px;padding:13px;font-size:13px;margin-bottom:12px}.error button{margin-left:12px;padding:6px 12px;background:white;border:1px solid #b58a77;border-radius:4px;color:#813b27}[hidden]{display:none!important}@media(max-width:1050px){.sidebar{width:230px;padding:26px 18px}.brand{font-size:22px}header{padding:22px 24px}.chat-area,.composer-wrap{padding-left:24px;padding-right:24px}.welcome h2{font-size:36px}.examples{gap:8px}.examples button{padding:14px 10px}}@media(max-width:720px){body{display:block}.sidebar{width:100%;height:auto;position:static;display:grid;grid-template-columns:1fr auto;gap:14px;padding:18px}.brand{font-size:20px}.brand-sub{font-size:7px}.mark{width:34px;height:34px}.side-section,.side-note,.side-footer,.demo-tag{display:none}.new-button{font-size:11px;padding:10px}.guide-link{margin:0;padding:0;border:0;grid-column:1/-1;font-size:11px;justify-content:flex-start;gap:6px}header{padding:18px}.private-label{display:none}h1{font-size:17px}.connection{padding:10px 18px}.chat-area{padding:18px}.welcome{padding-top:5px}.welcome h2{font-size:34px}.examples{grid-template-columns:1fr;margin:22px 0}.examples button{padding:13px 16px}.card-label{margin-bottom:5px}.composer-wrap{padding:12px 14px}.composer-bottom>span{font-size:9px}.message.assistant{margin-right:0}.welcome-icon{margin-bottom:16px}.composer-wrap{position:static}}

`````

## tests/browser-check.js

`````
async (page) => {
  // Browser-only API fixtures: never send a chat to the real model in this check.
  let mode = 'success';
  const sent = [];
  await page.route('**/api/chat', async route => {
    sent.push(route.request().postDataJSON());
    if (mode === 'offline') return route.abort('failed');
    await new Promise(resolve => setTimeout(resolve, 150));
    await route.fulfill({json: {
      answer: 'Browser test fixture [1]\n<img src=x onerror=alert(1)>\nportal.hospital.example',
      sources: [
        {label: 1, filename: 'fixture.txt', section: 'Test', subsection: 'Cited', cited: true},
        {label: 2, filename: 'fixture.txt', section: 'Other', subsection: '', cited: false},
      ], search_used: true, elapsed_seconds: 0.15,
    }});
  });
  await page.goto('http://127.0.0.1:8000');
  await page.getByText('Connected to the hospital RAG pipeline.').waitFor();
  await page.setViewportSize({width: 1440, height: 1000});
  await page.screenshot({path: 'output/playwright/desktop.png', fullPage: true});
  await page.getByRole('button', {name: /FIND A DEPARTMENT/}).click();
  const input = page.getByRole('textbox', {name: 'Your question'});
  if (await input.inputValue() !== 'Where is the cardiology department?') throw Error('Example did not fill input');
  await input.press('Shift+Enter');
  if (!(await input.inputValue()).includes('\n')) throw Error('Shift+Enter failed');
  await input.press('Enter');
  await page.getByText('References · 1 cited / 2 retrieved').waitFor();
  await page.getByText('References · 1 cited / 2 retrieved').click();
  await page.getByText('[2] fixture.txt · Retrieved only').waitFor();
  if (await page.locator('#messages img').count()) throw Error('Unsafe HTML rendered');
  if (await page.locator('#messages a').count()) throw Error('Fictional address became a link');
  const firstThread = sent[0].conversation_id;
  mode = 'offline';
  await input.fill('Retry test');
  await input.press('Enter');
  await page.getByText('Cannot reach the local backend.', {exact: false}).waitFor();
  const count = await page.locator('.message.user').count();
  const requestId = sent[sent.length - 1].request_id;
  mode = 'success';
  await page.getByRole('button', {name: 'Retry', exact: true}).click();
  await page.waitForFunction(() => document.querySelectorAll('.message.assistant').length === 2);
  if (await page.locator('.message.user').count() !== count) throw Error('Retry duplicated user message');
  if (sent[sent.length - 1].request_id !== requestId) throw Error('Retry changed request ID');
  await page.getByRole('button', {name: '＋ New Conversation', exact: true}).click();
  await page.getByText('New conversation ready.', {exact: false}).waitFor();
  if (await page.locator('.message').count()) throw Error('Reset did not clear messages');
  await input.fill('New thread test');
  await input.press('Enter');
  await page.getByText('References · 1 cited / 2 retrieved').waitFor();
  if (sent[sent.length - 1].conversation_id === firstThread) throw Error('Reset reused old ID');
  await page.setViewportSize({width: 390, height: 844});
  await page.screenshot({path: 'output/playwright/mobile.png', fullPage: true});
  if (await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)) throw Error('Mobile horizontal overflow');
  await page.goto('http://127.0.0.1:8000/guide/');
  await page.locator('img').waitFor();
  if (!await page.locator('img').evaluate(img => img.complete && img.naturalWidth > 0)) throw Error('Diagram image failed');
  await page.setViewportSize({width: 1200, height: 1100});
  await page.screenshot({path: 'output/playwright/guide.png', fullPage: true});
  console.log('PASS: examples, Enter/Shift+Enter, references, safe output, disconnected retry, new ID, mobile width, diagram image. Browser chat responses were fixtures; no model calls.');
}

`````

## tests/test_backend.py

`````
import asyncio
import threading
import time
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from backend.main import create_app


class FakeGraph:
    def __init__(self, delay=0, fail=False):
        self.calls = []
        self.delay = delay
        self.fail = fail
        self.active = set()
        self.overlap = False
        self.lock = threading.Lock()

    def invoke(self, state, config):
        thread = config['configurable']['thread_id']
        with self.lock:
            self.overlap |= thread in self.active
            self.active.add(thread)
            self.calls.append((thread, state['question']))
        time.sleep(self.delay)
        with self.lock:
            self.active.remove(thread)
        if self.fail:
            raise RuntimeError('PRIVATE_KEY and /private/internal/path must never reach the browser')
        return {'answer': 'Test answer [1]', 'searched': True, 'sources': [
            {'label': 1, 'filename': 'test.txt', 'section': 'Test', 'subsection': 'Details'},
            {'label': 2, 'filename': 'test.txt', 'section': 'Other', 'subsection': ''}]}


def payload(conversation_id, question='Hello'):
    return {'conversation_id': conversation_id, 'question': question, 'request_id': str(uuid4())}


def test_response_isolation_retry_and_validation():
    graph = FakeGraph()
    with TestClient(create_app(graph=graph)) as client:
        assert client.get('/api/health').json()['ready']
        first = client.post('/api/conversations').json()['conversation_id']
        second = client.post('/api/conversations').json()['conversation_id']
        request = payload(first)
        response = client.post('/api/chat', json=request)
        assert response.status_code == 200
        data = response.json()
        assert data['answer'] == 'Test answer [1]'
        assert data['search_used'] is True
        assert data['elapsed_seconds'] >= 0
        assert [s['cited'] for s in data['sources']] == [True, False]
        assert client.post('/api/chat', json=request).json() == data
        assert len(graph.calls) == 1
        assert client.post('/api/chat', json=payload(second)).status_code == 200
        assert graph.calls[0][0] != graph.calls[1][0]
        for question in [' ', 'x' * 2001]:
            assert client.post('/api/chat', json=payload(first, question)).status_code == 422
        assert client.post('/api/chat', json=payload('invalid')).status_code == 422
        assert client.post('/api/chat', json=payload(str(uuid4()))).status_code == 404
        request['question'] = 'Changed'
        assert client.post('/api/chat', json=request).status_code == 409
        assert client.post('/api/conversations', headers={'Origin': 'https://other.example'}).status_code == 403
        assert client.get('/.env').status_code == 404


def test_safe_error_and_failed_conversation():
    graph = FakeGraph(fail=True)
    with TestClient(create_app(graph=graph)) as client:
        conversation = client.post('/api/conversations').json()['conversation_id']
        response = client.post('/api/chat', json=payload(conversation))
        assert response.status_code == 502
        assert 'PRIVATE_KEY' not in response.text
        assert '/private/' not in response.text
        client.post('/api/chat', json=payload(conversation, 'Again'))
        assert len(graph.calls) == 1


def test_timeout_retry_does_not_repeat_graph_call():
    graph = FakeGraph(delay=.08)
    with TestClient(create_app(graph=graph, timeout=.02)) as client:
        conversation = client.post('/api/conversations').json()['conversation_id']
        request = payload(conversation)
        assert client.post('/api/chat', json=request).status_code == 504
        time.sleep(.12)
        assert client.post('/api/chat', json=request).status_code == 200
        assert len(graph.calls) == 1


def test_same_conversation_requests_are_serialized():
    async def check():
        graph = FakeGraph(delay=.04)
        app = create_app(graph=graph)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
                identifier = (await client.post('/api/conversations')).json()['conversation_id']
                results = await asyncio.gather(*[
                    client.post('/api/chat', json=payload(identifier, question))
                    for question in ['First', 'Second']])
                assert all(result.status_code == 200 for result in results)
                assert len(graph.calls) == 2
                assert not graph.overlap
    asyncio.run(check())


def test_actionable_startup_error(monkeypatch):
    from backend import main
    from backend.rag_adapter import SetupError
    def missing():
        raise SetupError('Hospital index is missing. Run the index builder first.')
    monkeypatch.setattr(main, 'load_graph', missing)
    with TestClient(create_app()) as client:
        assert client.get('/').status_code == 200
        assert client.get('/api/health').json()['ready'] is False
        assert client.post('/api/conversations').status_code == 503


def test_adapter_missing_path(monkeypatch):
    import pytest
    from backend import rag_adapter
    monkeypatch.setattr(rag_adapter, 'load_dotenv', lambda *args: None)
    monkeypatch.delenv('RAG_PROJECT_PATH', raising=False)
    with pytest.raises(rag_adapter.SetupError, match='Set RAG_PROJECT_PATH'):
        rag_adapter.load_graph()


def test_adapter_missing_index(monkeypatch, tmp_path):
    import pytest
    from backend import rag_adapter
    source = tmp_path / 'src/agents'
    source.mkdir(parents=True)
    (source / 'agent_graph.py').touch()
    monkeypatch.setattr(rag_adapter, 'load_dotenv', lambda *args: None)
    monkeypatch.setenv('RAG_PROJECT_PATH', str(tmp_path))
    monkeypatch.setenv('OPENAI_API_KEY', 'test-placeholder-not-a-real-key')
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    with pytest.raises(rag_adapter.SetupError, match='index is missing'):
        rag_adapter.load_graph()

`````

## docs/RUNBOOK.md

`````
# Run the hospital UI locally

## 1. Open the right folder

Place `hospital-ui` beside `RAG-Basic-project-Hospital-`. All commands below run from `hospital-ui` unless stated otherwise. This is a separate local Git repository; no remote is required.

```bash
cd /path/to/hospital-testing/hospital-ui
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r ../RAG-Basic-project-Hospital-/requirements.txt
```

The original workspace uses Python 3.14.3. Install both dependency files into this UI environment so the imported graph has all its libraries. Do not copy the original environment or Chroma files.

## 2. Configure the connection

For a first setup only:

```bash
cp .env.example .env
```

Edit `.env` in your editor:

```dotenv
RAG_PROJECT_PATH=../RAG-Basic-project-Hospital-
CHAT_TIMEOUT_SECONDS=120
```

An absolute path such as `/path/to/hospital-testing/RAG-Basic-project-Hospital-` also works. Relative paths resolve from `hospital-ui`, regardless of where the server command was launched.

The original RAG `.env` supplies `OPENAI_API_KEY` and optional `LANGSMITH_*` settings. Any backend override belongs in the UI `.env`, never in JavaScript or HTML. Do not share keys in chat, screenshots, Git, or error reports. Verify exclusion with `git check-ignore .env`.

## 3. Check the original index

If the original app already works and has a populated `hospital_info` collection, skip building. The adapter uses the original `chroma_db/chroma.sqlite3` in place and checks that the collection has records.

If the index is missing, run the existing builder from the original repository using the UI environment:

```bash
cd ../RAG-Basic-project-Hospital-
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

Use `python -m src.evaluation.evaluate` from the original RAG repository for repeatable, saved human evaluation reports. The UI's response details and exploratory chats are not an evaluation report.

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

`````

## docs/VERIFICATION.md

`````
# Verification performed

- Installed both the UI and original RAG requirements into the UI's separate virtual environment. `pip check` reported no broken requirements.
- All 7 backend pytest checks passed with a mocked graph: response fields and citation status, distinct thread IDs, duplicate request reuse, question/ID validation, same-conversation serialization, timeout-and-retry behavior, sanitized graph errors, and missing-path/index startup behavior.
- Started the real adapter and obtained `ready: true` from `/api/health`. This initialized the real LangGraph graph and inspected the original Chroma database without requesting an answer.
- Browser checks ran in Chromium through Playwright. The browser used explicit intercepted chat fixtures, while the real local server served the UI and issued conversation IDs. Verified examples, Enter/Shift+Enter, references, HTML-like text safety, plain-text fictional addresses, connection failure and retry, fresh IDs, cleared visible conversation, mobile width, and successful diagram loading.
- Inspected desktop and diagram screenshots. Artifacts are under ignored `output/playwright/`.
- Initialized a separate local Git repository. No remote, commits, or pushes were created. `.env` and `.venv/` are ignored.

No paid OpenAI calls, live answer tests, or full RAG evaluations were run. Correct factual answers and semantic follow-up behavior need the manual checks in RUNBOOK.md. No screen-reader audit or public deployment testing was performed.

The authored SVG is directly viewable in a browser and does not require a Mermaid renderer. The accompanying Mermaid text is an editable alternate representation and was successfully validated by rendering it with Mermaid CLI into an ignored output artifact. The renderer was a verification tool only; the app has no Node runtime dependency.

`````

## docs/flowchart.mmd

`````
flowchart TD
    subgraph indexing["1. Build index separately"]
      document["Hospital document"] --> loader["Load text and source"]
      loader --> chunks["Split at headings"] --> embeddings["OpenAI: embed new chunks"]
      embeddings --> saved[("Persisted Chroma index")]
    end
    subgraph chatting["2. Chat using the saved index"]
      browser["Browser question"] --> backend["Local FastAPI backend"]
      backend --> history["LangGraph conversation state"] --> decision{"AI requests search?"}
      decision -->|No| direct["Greeting, clarification, or scope reply"]
      decision -->|Yes| query["Focused query and query embedding"]
      query --> search["Search saved Chroma index"] --> passages["Current-turn passages"]
      passages --> answer["Same LLM writes cited answer"]
      direct --> display["Browser displays answer and references"]
      answer --> display
    end

`````

## docs/flowchart.svg

`````
<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1160" viewBox="0 0 1000 1160" role="img" aria-labelledby="title desc">
<title id="title">Hospital assistant: prepare once, then ask questions</title><desc id="desc">Index building is separate from chatting. The model chooses a direct reply or one hospital search. Query embeddings search the saved Chroma index, and the model generates a cited answer.</desc>
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#527064"/></marker></defs>
<rect width="1000" height="1160" fill="#fafbf8"/>
<g font-family="Arial,sans-serif" text-anchor="middle" fill="#203d37">
<text x="500" y="47" font-size="26" font-weight="bold">How your hospital assistant works</text>
<text x="500" y="76" font-size="15">Follow the arrows from top to bottom</text>
<rect x="24" y="105" width="310" height="895" rx="16" fill="#f0f4ee" stroke="#d1ddcc"/>
<rect x="355" y="105" width="621" height="895" rx="16" fill="white" stroke="#d1ddcc"/>
<text x="179" y="144" font-size="20" font-weight="bold">1. Prepare the index</text>
<text x="179" y="172" font-size="13">Run separately before chatting</text>
<text x="665" y="144" font-size="20" font-weight="bold">2. Ask a question</text>
<text x="665" y="172" font-size="13">Reuse the saved hospital index</text>
<g fill="white" stroke="#b6c7bb"><rect x="55" y="210" width="248" height="70" rx="10"/><rect x="55" y="330" width="248" height="70" rx="10"/><rect x="55" y="450" width="248" height="70" rx="10"/></g>
<g font-size="16"><text x="179" y="241">Hospital document</text><text x="179" y="263" font-size="12">Existing fictional hospital text</text><text x="179" y="362">Load the document</text><text x="179" y="384" font-size="12">Keep its source filename</text><text x="179" y="481">Split into passages</text><text x="179" y="504" font-size="12">Follow section headings</text></g>
<rect x="55" y="570" width="248" height="70" rx="10" fill="#e3effb" stroke="#7595b4"/><text x="179" y="600" font-size="16">Embed new passages</text><text x="179" y="624" font-size="12">OpenAI turns text into numbers</text>
<rect x="55" y="700" width="248" height="95" rx="15" fill="#e4edda" stroke="#7e9862"/><text x="179" y="736" font-size="17">Save to Chroma</text><text x="179" y="760" font-size="13">Hospital passages + vectors</text><text x="179" y="780" font-size="12">Stored in the original chroma_db/</text>
<g stroke="#527064" stroke-width="2" fill="none" marker-end="url(#arrow)"><path d="M179 280V323"/><path d="M179 400V443"/><path d="M179 520V563"/><path d="M179 640V693"/></g>
<text x="179" y="858" font-size="14">Build again when the document</text><text x="179" y="880" font-size="14">changes. Unchanged chunks</text><text x="179" y="902" font-size="14">reuse their saved embeddings.</text>
<rect x="515" y="210" width="300" height="60" rx="10" fill="#fff" stroke="#b6c7bb"/><text x="665" y="245" font-size="16">Type a question in the browser</text>
<rect x="515" y="307" width="300" height="66" rx="10" fill="#fff0db" stroke="#c79b62"/><text x="665" y="335" font-size="16">Add conversation history</text><text x="665" y="357" font-size="12">Helps with “its hours” or “there”</text>
<path d="M665 406L791 477L665 548L539 477Z" fill="#e3effb" stroke="#7595b4"/><text x="665" y="470" font-size="16">AI decision:</text><text x="665" y="492" font-size="16">search needed?</text>
<rect x="391" y="586" width="205" height="83" rx="10" fill="#e3effb" stroke="#7595b4"/><text x="493" y="615" font-size="16">Reply directly</text><text x="493" y="637" font-size="12">Greeting, clarification,</text><text x="493" y="655" font-size="12">or scope response</text>
<rect x="646" y="586" width="285" height="70" rx="10" fill="#e3effb" stroke="#7595b4"/><text x="788" y="616" font-size="16">Focused query → embedding</text><text x="788" y="640" font-size="12">AI writes the query; embeddings match text</text>
<rect x="646" y="697" width="285" height="70" rx="10" fill="#e4edda" stroke="#7e9862"/><text x="788" y="726" font-size="16">Search saved Chroma index</text><text x="788" y="751" font-size="12">Return up to six hospital passages</text>
<rect x="646" y="808" width="285" height="70" rx="10" fill="#e3effb" stroke="#7595b4"/><text x="788" y="837" font-size="16">AI writes a cited answer</text><text x="788" y="861" font-size="12">Use this turn’s retrieved passages</text>
<rect x="515" y="921" width="360" height="54" rx="27" fill="#24594b"/><text x="695" y="954" font-size="16" fill="white">Show answer and available references</text>
<g stroke="#527064" stroke-width="2" fill="none" marker-end="url(#arrow)"><path d="M665 270V300"/><path d="M665 373V399"/><path d="M539 477H493V579"/><path d="M791 477H845V579"/><path d="M788 656V690"/><path d="M788 767V801"/><path d="M788 878V914"/><path d="M493 669V948H508"/></g>
<text x="476" y="537" font-size="14">No</text><text x="866" y="537" font-size="14">Yes</text>
<g text-anchor="start" font-size="13"><rect x="60" y="1040" width="16" height="16" fill="#e4edda" stroke="#7e9862"/><text x="85" y="1053">Local disk storage</text><rect x="305" y="1040" width="16" height="16" fill="#fff0db" stroke="#c79b62"/><text x="330" y="1053">Temporary memory</text><rect x="565" y="1040" width="16" height="16" fill="#e3effb" stroke="#7595b4"/><text x="590" y="1053">External OpenAI services</text></g>
<text x="500" y="1100" font-size="13">New Conversation starts a fresh ID. Server restart clears conversation memory, not Chroma.</text><text x="500" y="1128" font-size="13">Optional LangSmith records traces. It does not provide hospital answers.</text>
</g></svg>

`````

## docs/guide.css

`````
body{margin:0;background:#fafbf8;color:#203d37;font:16px/1.8 Arial,sans-serif}main{max-width:1000px;margin:auto;padding:40px 24px}h1{font:42px/1.2 Georgia,serif}h2{font:28px Georgia,serif;margin-top:40px}.eyebrow{font-size:11px;letter-spacing:2px;margin-top:36px}img{display:block;width:100%;height:auto;background:white;border:1px solid #dce4df;border-radius:16px;margin:30px 0}a{color:#24594b}li{margin:12px 0}a:focus-visible{outline:3px solid #ba771f}

`````

## docs/index.html

`````
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>How the hospital assistant works</title><link rel="stylesheet" href="/guide/guide.css"></head>
<body><main><a href="/">← Back to the assistant</a><p class="eyebrow">A SIMPLE GUIDE</p><h1>Two processes. One helpful answer.</h1><p>First prepare the hospital document. Then ask questions using that saved information.</p><img src="/guide/flowchart.svg" alt="Two flowcharts: build the index from documents through chunks and embeddings to Chroma; chat uses history and an AI decision to either reply directly or search Chroma and generate a cited answer."><h2>Follow a question</h2><ol><li>The browser sends your question to the local Python backend.</li><li>LangGraph adds it to your conversation. The AI reads your history to understand follow-ups.</li><li>The AI may reply with a greeting, clarification question, or scope explanation. There is no separate clarification node.</li><li>For hospital facts, the AI writes a focused search query. The embedding model converts it into numbers, and Chroma finds up to six passages.</li><li>The same chat model writes an answer using the current question, resolved tool call, and current retrieved passages. Older passages are excluded at this stage.</li><li>The page displays the answer and references. Citation numbers restart for every response.</li></ol><h2>What is saved?</h2><p><strong>Chroma</strong> is the hospital index on disk. <strong>Conversation memory</strong> exists only in the Python server process. New Conversation selects a fresh ID; restarting the server clears all conversation state.</p><p><strong>Embeddings</strong> find similar text. The <strong>LLM</strong> decides whether to search and writes answers. Optional <strong>LangSmith</strong> traces record activity for debugging; they do not supply hospital answers.</p><p>The real graph is prepare → agent → optional search → agent → finish. A hospital question uses at most one search. Evaluation remains in the original project and saves reports; exploring this chat does not replace those repeatable tests.</p><p><a href="/guide/flowchart.svg">Open the full-size diagram</a> · <a href="/guide/RUNBOOK.md">Local runbook</a></p></main></body></html>

`````
