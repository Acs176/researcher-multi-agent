# Researcher Multi-Agent (Semantic Kernel + Python)

Two-agent, sequential flow built on Semantic Kernel's agent orchestration.

## Architecture
- Planner: decides if clarification is needed and proposes search queries.
- Searcher: runs searches and returns structured results.

Coordination uses Semantic Kernel's sequential orchestration (actor model), not a custom bus.

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
python main.py "..." --docs-root . --search local --timeout 120 --show-steps
```

Logging:
```
python main.py "..." --log-level DEBUG
```

Azure OpenAI (optional):
```
export AZURE_OPENAI_ENDPOINT="https://...openai.azure.com/"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_DEPLOYMENT="my-deployment"
```

Brave web search (Semantic Kernel connector):
```
export BRAVE_API_KEY="..."
python main.py "..." --search brave
```

## Notes
- Search is local-only (simple file scan) by default. Use the Brave connector for web search.
- `--show-steps` prints each agent output as it completes.
- All agent prompts live in `prompts.py`.
