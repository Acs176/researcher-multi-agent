# Researcher Multi-Agent (Semantic Kernel + Python)

Single-agent flow built on Semantic Kernel's agent orchestration.

## Architecture
- Researcher: decides when to search, calls the search tool, and returns a summarized answer.

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

## Eval runner
Run a queryset with LangGraph and persist per-query logs (JSONL):
```
python eval_runner.py --queryset eval/queryset.json --eval-id eval_001 --eval-log eval/eval_runs.jsonl
```
Common filters:
```
python eval_runner.py --bucket freshness --limit 5 --search brave
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
