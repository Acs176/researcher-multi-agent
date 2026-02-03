# Researcher Multi-Agent (LangGraph + Python)

Multi-agent flow built on LangGraph with a planner → searcher → fetcher → summarizer pipeline.

## Architecture
- Planner: decides whether to search and proposes queries.
- Searcher: executes searches and returns results.
- URL selector: picks the most relevant URLs to fetch.
- Fetcher: retrieves content from selected URLs.
- Summarizer: answers the question from fetched content (or falls back when none).

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
python main.py "..." --search brave --timeout 120 --show-steps
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

Brave web search (Brave Search API):
```
export BRAVE_API_KEY="..."
python main.py "..." --search brave
```

## Notes
- Search uses the Brave API when `--search brave` is selected.
- `--show-steps` prints each agent output as it completes.
- All agent prompts live in `prompts.py`.
- The LangGraph pipeline lives in `langgraph_flow.py`.
