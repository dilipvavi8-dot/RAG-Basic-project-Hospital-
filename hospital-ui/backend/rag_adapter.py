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
        raise SetupError('RAG_PROJECT_PATH must point to the rag-pipeline folder containing src/ and chroma_db/.')
    load_dotenv(project / '.env')
    if not os.getenv('OPENAI_API_KEY', '').strip():
        raise SetupError('OPENAI_API_KEY is missing or empty. Check backend environment configuration and restart.')
    if os.getenv('LANGSMITH_TRACING', '').lower() == 'true' and not os.getenv('LANGSMITH_API_KEY', '').strip():
        raise SetupError('LangSmith tracing is enabled without a key. Set LANGSMITH_API_KEY or disable tracing, then restart.')
    database = project / 'chroma_db/chroma.sqlite3'
    if not database.is_file():
        raise SetupError('Hospital index is missing. Run python -m scripts.build_index from rag-pipeline/ first.')
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
