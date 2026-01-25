# Researcher Multi-Agent (Semantic Kernel + Python)

Decentralized, message-passing team of AI agents that collaboratively answer complex research questions.

## Architecture
- Planner: breaks the user query into sub-tasks and proposes search queries.
- Search: runs local doc search (RAG-style) and returns raw snippets.
- Summarizer: condenses raw snippets into cited bullets.
- Critic/Verifier: checks gaps/contradictions and requests more info if needed.
- Writer: produces the final response using the verified summary.

Coordination is decentralized: agents subscribe to message types on a shared bus and publish follow-up messages.

## Setup
```
pip install -r requirements.txt
export OPENAI_API_KEY="..."
export SK_MODEL="gpt-5-mini"
```

## Run
```
python main.py "What are the latest trends in retrieval-augmented generation?"
```

Optional args:
```
python main.py "..." --docs-root . --search local --timeout 120
```

Logging:
```
python main.py "..." --log-level DEBUG
```

Tuning (env vars):
- `SK_MAX_TOKENS` (default: 900)
- `SK_MAX_COMPLETION_TOKENS` (optional override)
- `SK_REASONING_EFFORT` (recommended for gpt-5: `low`)

Brave web search (Semantic Kernel connector):
```
export BRAVE_API_KEY="..."
python main.py "..." --search brave
```

## Notes
- Search is local-only (simple file scan) by default. Use the Brave connector for web search.
- `info_request` messages are only routed to Search when `intent="search"` to avoid accidental web queries.
- All agent prompts live in `prompts.py`.
