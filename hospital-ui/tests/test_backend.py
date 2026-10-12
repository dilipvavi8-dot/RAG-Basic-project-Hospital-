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
