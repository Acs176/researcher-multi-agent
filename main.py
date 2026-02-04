from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

from dotenv import load_dotenv

from search_plugin import SearchPlugin
from eval_logging import RunLogger
from langgraph_flow import build_graph, build_openai_client
from otel_setup import init_tracing
from search_providers import BraveSearchProvider, NoopSearchProvider


def _build_provider(mode: str, brave_key: str | None):
    """Construct the CLI search provider."""
    if mode == "brave":
        return BraveSearchProvider(api_key=brave_key)
    return NoopSearchProvider()


async def run() -> None:
    """CLI entrypoint for a single LangGraph query."""
    parser = argparse.ArgumentParser(description="LangGraph multi-agent research coordinator")
    parser.add_argument("query", nargs="*", help="Research question")
    parser.add_argument("--search", default="brave", choices=["none", "brave"])
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    parser.add_argument("--show-steps", action="store_true", help="Print agent outputs as they complete")
    args = parser.parse_args()

    load_dotenv()
    init_tracing()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    query = " ".join(args.query).strip()
    if not query:
        query = input("Research question: ").strip()

    brave_api_key = os.getenv("BRAVE_API_KEY")
    provider = _build_provider(args.search, brave_api_key)
    client, model = build_openai_client()

    run_logger = RunLogger(eval_id="cli", log_path="eval/eval_runs.jsonl")
    run_logger.record_query(query)
    search_plugin = SearchPlugin(provider, run_logger=run_logger)

    app = build_graph(
        client=client,
        model=model,
        search_plugin=search_plugin,
        run_logger=run_logger,
        timeout=args.timeout,
        show_steps=args.show_steps,
    )

    result = await app.ainvoke(
        {
            "query": query,
            "plan": {},
            "search_results": [],
            "urls": [],
            "fetched": [],
            "answer": "",
        }
    )
    answer = (result.get("answer") or "").strip()
    if not answer:
        answer = json.dumps(result.get("fetched", []), ensure_ascii=True)
    if not args.show_steps:
        print(answer)


if __name__ == "__main__":
    asyncio.run(run())
