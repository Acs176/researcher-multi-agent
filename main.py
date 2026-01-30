from __future__ import annotations

import argparse
import asyncio
import logging
import os

from orchestrator import ResearchCoordinator
from search_providers import (
    LocalDocSearchProvider,
    NoopSearchProvider,
    SemanticKernelBraveSearchProvider,
)
from sk_client import SKClient, config_from_env
from dotenv import load_dotenv
from otel_setup import init_tracing


def build_provider(mode: str, docs_root: str, brave_key: str | None):
    if mode == "local":
        return LocalDocSearchProvider(docs_root)
    if mode == "brave":
        return SemanticKernelBraveSearchProvider(api_key=brave_key)
    return NoopSearchProvider()


async def run() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent research coordinator")
    parser.add_argument("query", nargs="*", help="Research question")
    parser.add_argument("--docs-root", default=os.getcwd(), help="Root for local doc search")
    parser.add_argument("--search", default="local", choices=["local", "none", "brave"])
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
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
    provider = build_provider(args.search, args.docs_root, brave_api_key)
    llm = SKClient(config_from_env())
    coordinator = ResearchCoordinator(llm=llm, search_provider=provider)
    answer = await coordinator.run(query, timeout=args.timeout)
    print(answer)


if __name__ == "__main__":
    asyncio.run(run())
