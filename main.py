from __future__ import annotations

import argparse
import asyncio
import logging
import os
from semantic_kernel.agents import SequentialOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents.chat_message_content import ChatMessageContent

from agents import build_agents, build_kernel
from search_providers import (
    LocalDocSearchProvider,
    NoopSearchProvider,
    SemanticKernelBraveSearchProvider,
)
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
    provider = build_provider(args.search, args.docs_root, brave_api_key)
    kernel = build_kernel()
    members = build_agents(kernel, provider)

    async def _agent_response_callback(message):
        if args.show_steps:
            print("\n--- Agent Output ---")
            print(_extract_text(message))

    orchestration = SequentialOrchestration(
        members=members,
        agent_response_callback=_agent_response_callback if (args.show_steps) else None,
    )
    runtime = InProcessRuntime()
    runtime.start()
    result = await orchestration.invoke(task=query, runtime=runtime)
    final = await result.get(timeout=args.timeout)
    await runtime.stop_when_idle()
    answer = _extract_text(final)
    print(answer)


def _extract_text(message) -> str:
    if isinstance(message, list):
        parts = [_extract_text(item) for item in message]
        return "\n".join([part for part in parts if part])
    if isinstance(message, ChatMessageContent):
        return message.content or ""
    return str(message)


if __name__ == "__main__":
    asyncio.run(run())
