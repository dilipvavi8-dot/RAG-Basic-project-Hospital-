# Verification performed

- Installed both the UI and original RAG requirements into the UI's separate virtual environment. `pip check` reported no broken requirements.
- All 7 backend pytest checks passed with a mocked graph: response fields and citation status, distinct thread IDs, duplicate request reuse, question/ID validation, same-conversation serialization, timeout-and-retry behavior, sanitized graph errors, and missing-path/index startup behavior.
- Started the real adapter and obtained `ready: true` from `/api/health`. This initialized the real LangGraph graph and inspected the original Chroma database without requesting an answer.
- Browser checks ran in Chromium through Playwright. The browser used explicit intercepted chat fixtures, while the real local server served the UI and issued conversation IDs. Verified examples, Enter/Shift+Enter, references, HTML-like text safety, plain-text fictional addresses, connection failure and retry, fresh IDs, cleared visible conversation, mobile width, and successful diagram loading.
- Inspected desktop and diagram screenshots. Artifacts are under ignored `output/playwright/`.
- Initialized a separate local Git repository. No remote, commits, or pushes were created. `.env` and `.venv/` are ignored.

No paid OpenAI calls, live answer tests, or full RAG evaluations were run. Correct factual answers and semantic follow-up behavior need the manual checks in RUNBOOK.md. No screen-reader audit or public deployment testing was performed.

The authored SVG is directly viewable in a browser and does not require a Mermaid renderer. The accompanying Mermaid text is an editable alternate representation and was successfully validated by rendering it with Mermaid CLI into an ignored output artifact. The renderer was a verification tool only; the app has no Node runtime dependency.
