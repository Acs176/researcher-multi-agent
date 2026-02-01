from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

from polite_fetcher import FetchResult, PoliteFetcher
from prompts import PLANNER_SYSTEM, SEARCHER_SYSTEM
from search_providers import SearchProvider, SearchResult


class SearchPlugin:
    def __init__(self, provider: SearchProvider) -> None:
        self._provider = provider
        self._fetcher: Optional[PoliteFetcher] = None
        if _env_flag("SEARCH_FETCH_PAGES", default=False):
            self._fetcher = PoliteFetcher(
                min_interval=_env_float("SEARCH_FETCH_MIN_INTERVAL", 1.0),
                max_concurrency=_env_int("SEARCH_FETCH_MAX_CONCURRENCY", 2),
                max_retries=_env_int("SEARCH_FETCH_MAX_RETRIES", 2),
                user_agent=os.getenv(
                    "SEARCH_FETCH_USER_AGENT",
                    "researcher-multi-agent/1.0 (+contact@example.com)",
                ),
                max_text_chars=_env_int("SEARCH_FETCH_MAX_CHARS", 2000),
                max_html_chars=_env_int("SEARCH_FETCH_MAX_HTML_CHARS", 200_000),
                robots_ttl_seconds=_env_int("SEARCH_FETCH_ROBOTS_TTL", 86_400),
                trace_content=_env_flag("SEARCH_FETCH_TRACE_CONTENT", default=True),
                trace_content_max_chars=_env_int("SEARCH_FETCH_TRACE_MAX_CHARS", 0),
            )

    @kernel_function(name="search", description="Search for information by query and return JSON results.")
    async def search(self, query: str, limit: int = 5) -> str:
        results = await self._provider.search(query, limit=limit)
        items = [_result_to_dict(r) for r in results]
        for item in items:
            item["query"] = query
        payload: Dict[str, Any] = {"query": query, "results": items}
        if self._fetcher:
            urls = [r.url for r in results if _should_fetch_url(r.url)]
            fetched = await self._fetcher.fetch_many(urls)
            _apply_fetch_results(payload["results"], fetched)
        return json.dumps(payload, ensure_ascii=True)


def build_kernel() -> Kernel:
    kernel = Kernel()
    service = _build_chat_service()
    kernel.add_service(service)
    return kernel


def build_agents(kernel: Kernel, provider: SearchProvider) -> List[ChatCompletionAgent]:
    planner = ChatCompletionAgent(
        name="Planner",
        instructions=PLANNER_SYSTEM,
        kernel=kernel,
    )
    searcher = ChatCompletionAgent(
        name="Search",
        instructions=SEARCHER_SYSTEM,
        kernel=kernel,
        plugins=[SearchPlugin(provider)],
    )
    return [planner, searcher]


def _build_chat_service():
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if azure_endpoint and azure_key and azure_deployment:
        return AzureChatCompletion(
            deployment_name=azure_deployment,
            endpoint=azure_endpoint,
            api_key=azure_key,
        )
    model = os.getenv("SK_MODEL", "gpt-5-mini")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    org_id = os.getenv("OPENAI_ORG_ID")
    return OpenAIChatCompletion(
        service_id="chat",
        ai_model_id=model,
        api_key=api_key,
        org_id=org_id,
    )


def _result_to_dict(result: SearchResult) -> Dict[str, Any]:
    return {"title": result.title, "snippet": result.snippet, "url": result.url}


def _apply_fetch_results(results: List[Dict[str, Any]], fetched: Dict[str, FetchResult]) -> None:
    for res in results:
        url = res.get("url")
        if not url:
            continue
        fetch = fetched.get(url)
        if not fetch:
            continue
        res["content"] = fetch.text
        res["fetch_status"] = fetch.status
        res["robots_allowed"] = fetch.robots_allowed
        if fetch.error:
            res["fetch_error"] = fetch.error


def _should_fetch_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
