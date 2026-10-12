"""Opt-in live API checks. Uses configured APIs and can incur charges.

Run from hospital-ui: .venv/bin/python tests/live_validation.py
This is deliberately not discovered by pytest.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx


def main():
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root.parent / 'rag-pipeline/data/sample_queries/questions.json').read_text())
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'mode': 'real_ui_api',
              'results': [], 'status': 'running'}
    destination = root / 'output/live_validation.json'
    destination.parent.mkdir(exist_ok=True)
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=180) as client:
        health = client.get('/api/health').json()
        if not health['ready']:
            raise SystemExit('Backend not ready. Inspect /api/health before live testing.')
        for case in cases:
            conversation = client.post('/api/conversations', json={}).json()['conversation_id']
            for step in case.get('steps', [case]):
                if step.get('reset_before'):
                    conversation = client.post('/api/conversations', json={}).json()['conversation_id']
                response = client.post('/api/chat', json={
                    'question': step['question'], 'conversation_id': conversation,
                    'request_id': str(uuid4()),
                })
                data = response.json()
                entry = {'case_id': case['id'], 'step_id': step['id'],
                         'question': step['question'], 'status_code': response.status_code,
                         'expected_search': step.get('expected_search'),
                         'expected_behavior': step['expected_behavior'],
                         'expected_facts': step.get('expected_facts', []),
                         'must_not_claim': step.get('must_not_claim', []), 'response': data}
                if response.is_success:
                    labels = set(re.findall(r'\[(\d+)\]', data['answer']))
                    source_labels = {str(source['label']) for source in data['sources']}
                    entry['citation_labels_resolve'] = labels.issubset(source_labels)
                    entry['routing_match'] = (None if step.get('expected_search') is None
                                              else step['expected_search'] == data['search_used'])
                report['results'].append(entry)
                destination.write_text(json.dumps(report, indent=2, ensure_ascii=False))
                print(json.dumps(entry, ensure_ascii=False), flush=True)
                if not response.is_success:
                    report['status'] = 'stopped_on_api_error'
                    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False))
                    return
    report['status'] = 'completed'
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
