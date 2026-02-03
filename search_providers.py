from __future__ import annotations

import asyncio
import html
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import httpx


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


class SearchProvider:
    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Search for a query and return normalized results."""
        raise NotImplementedError


class NoopSearchProvider(SearchProvider):
    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Return no results for any query."""
        return []


class BraveSearchProvider(SearchProvider):
    """Web search provider backed by the Brave Search API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        min_interval: float = 1.1,  # brave free, limited at 1 call / sec
        max_retries: int = 2,
        backoff_base: float = 1.0,
        timeout: float = 15.0,
        endpoint: str = "https://api.search.brave.com/res/v1/web/search",
    ) -> None:
        if not api_key:
            raise RuntimeError("BRAVE_API_KEY is required for Brave search.")
        self._api_key = api_key
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._timeout = timeout
        self._endpoint = endpoint
        self._last_request_at = 0.0
        self._logger = logging.getLogger("search.brave")

    async def search(self, query: str, limit: int = 2) -> List[SearchResult]:
        """Search Brave and return normalized results."""
        if not query:
            return []
        return await self._run_search(query=query, limit=limit)

    async def _run_search(self, query: str, limit: int) -> List[SearchResult]:
        """Execute Brave search with retry/backoff."""
        attempts = 0
        while True:
            await self._throttle()
            try:
                return await self._call_search(query=query, limit=limit)
            except Exception as exc:  # pragma: no cover - runtime dependency
                if _is_rate_limit_error(exc):
                    if attempts < self._max_retries:
                        delay = self._backoff_base * (2**attempts) + random.uniform(0, 0.25)
                        self._logger.warning(
                            "Brave rate limited (429). Retrying in %.2fs (attempt %s/%s).",
                            delay,
                            attempts + 1,
                            self._max_retries,
                        )
                        await asyncio.sleep(delay)
                        attempts += 1
                        continue
                    self._logger.warning(
                        "Brave rate limited (429). Giving up after %s attempts.",
                        attempts + 1,
                    )
                    return []
                self._logger.warning(
                    "Brave search failed for query=%s error=%s", query, exc, exc_info=True
                )
                return []

    async def _call_search(self, query: str, limit: int) -> List[SearchResult]:
        """Call the Brave Search API and parse results."""
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        params = {"q": query, "count": str(limit)}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._endpoint, params=params, headers=headers)
        if resp.status_code == 429:
            raise httpx.HTTPStatusError(
                "Brave Search rate limited",
                request=resp.request,
                response=resp,
            )
        if resp.status_code >= 400:
            self._logger.warning(
                "Brave search failed status=%s body=%s",
                resp.status_code,
                (resp.text or "")[:200],
            )
            return []
        try:
            data = resp.json()
        except ValueError:
            self._logger.warning("Brave search returned invalid JSON.")
            return []
        return _parse_brave_results(data, limit=limit)

    async def _throttle(self) -> None:
        """Throttle requests to respect the minimum interval."""
        if self._min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_at
        wait_for = self._min_interval - elapsed
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        self._last_request_at = time.monotonic()


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if an exception indicates rate limiting."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) if response is not None else None
    if status is not None:
        return status == 429
    status = getattr(exc, "status_code", None)
    return status == 429


def _parse_brave_results(data: Any, limit: int) -> List[SearchResult]:
    """Parse Brave API JSON into SearchResult objects."""
    items: List[SearchResult] = []
    web = data.get("web") if isinstance(data, dict) else None
    results = web.get("results") if isinstance(web, dict) else None
    if not isinstance(results, list):
        return items
    for result in results:
        if not isinstance(result, dict):
            continue
        url = result.get("url") or result.get("link") or ""
        title = result.get("title") or ""
        snippet = result.get("description") or result.get("snippet") or ""
        if not url:
            continue
        items.append(
            SearchResult(
                title=title or (url[:60] + "..." if len(url) > 60 else url),
                snippet=_clean_snippet(snippet),
                url=url,
            )
        )
        if len(items) >= limit:
            break
    return items


_TAG_RE = re.compile(r"<[^>]+>")


def _clean_snippet(text: str) -> str:
    """Strip HTML tags and entities from snippets."""
    if not text:
        return ""
    return _TAG_RE.sub("", html.unescape(text)).strip()
