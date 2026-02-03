from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from eval_logging import RunLogger
from polite_fetcher import FetchResult, PoliteFetcher
from search_providers import SearchProvider, SearchResult


class SearchPlugin:
    """Adapter that executes provider search and optional fetching."""

    def __init__(self, provider: SearchProvider, run_logger: Optional["RunLogger"] = None) -> None:
        """Initialize with a provider and optional run logger."""
        self._provider = provider
        self._run_logger = run_logger
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

    async def search(self, query: str) -> str:
        """Search for a query and return a JSON payload string."""
        results = await self._provider.search(query, limit=2)  # Hardcoded for now
        items = [_result_to_dict(r) for r in results]
        for item in items:
            item["query"] = query
        payload: Dict[str, Any] = {"query": query, "results": items}
        if self._fetcher:
            urls = [r.url for r in results if _should_fetch_url(r.url)]
            fetched = await self._fetcher.fetch_many(urls)
            _apply_fetch_results(payload["results"], fetched)
        if self._run_logger:
            self._run_logger.record_search(query, payload["results"])
        return json.dumps(payload, ensure_ascii=True)


def _result_to_dict(result: SearchResult) -> Dict[str, Any]:
    """Convert a SearchResult to a serializable dict."""
    return {"title": result.title, "snippet": result.snippet, "url": result.url}


def _apply_fetch_results(results: List[Dict[str, Any]], fetched: Dict[str, FetchResult]) -> None:
    """Attach fetched page content to search results."""
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
    """Return True for http(s) URLs eligible for fetching."""
    return url.startswith("http://") or url.startswith("https://")


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment flag with a default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a default."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a default."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
