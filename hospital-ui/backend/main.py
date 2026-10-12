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
