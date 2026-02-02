from __future__ import annotations

import asyncio
import html
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Any


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str


class SearchProvider:
    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        raise NotImplementedError


class LocalDocSearchProvider(SearchProvider):
    """Naive local search across files for lightweight RAG over docs."""

    def __init__(self, root: str, include_ext: Iterable[str] | None = None) -> None:
        self.root = root
        self.include_ext = set(include_ext or [".md", ".txt", ".rst"])

    def _iter_files(self) -> Iterable[str]:
        for dirpath, _, filenames in os.walk(self.root):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in self.include_ext:
                    yield os.path.join(dirpath, name)

    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        results: List[SearchResult] = []
        q = query.lower().strip()
        if not q:
            return results
        for path in self._iter_files():
            if len(results) >= limit:
                break
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if q in line.lower():
                            results.append(
                                SearchResult(
                                    title=os.path.basename(path),
                                    snippet=line.strip()[:280],
                                    url=f"file://{path}",
                                )
                            )
                            break
            except OSError:
                continue
        return results


class NoopSearchProvider(SearchProvider):
    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        return []


class SemanticKernelBraveSearchProvider(SearchProvider):
    """Web search provider backed by Semantic Kernel's BraveSearch connector."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        min_interval: float = 1.1, # brave free, limited at 1 call / sec
        max_retries: int = 2,
        backoff_base: float = 1.0,
    ) -> None:
        try:
            from semantic_kernel.connectors.brave import BraveSearch
            from semantic_kernel.data.text_search import TextSearchResult
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Semantic Kernel BraveSearch connector is unavailable. "
                "Ensure semantic-kernel is installed with the required extras."
            ) from exc
        self._text_search_result_type = TextSearchResult
        self._client = BraveSearch(api_key=api_key) if api_key else BraveSearch()
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._last_request_at = 0.0
        self._logger = logging.getLogger("search.brave")

    async def search(self, query: str, limit: int = 2) -> List[SearchResult]:
        if not query:
            return []
        results = await self._run_search(query=query, limit=limit)
        return await _to_search_results(results)

    async def _run_search(self, query: str, limit: int):
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

    async def _call_search(self, query: str, limit: int):
        if hasattr(self._client, "search"):
            try:
                return await self._client.search(
                    query=query,
                    output_type=self._text_search_result_type,
                    top=limit,
                )
            except TypeError:
                return await self._client.search(query=query, top=limit)
        if hasattr(self._client, "get_text_search_results"):
            return await self._client.get_text_search_results(query=query, top=limit)
        raise RuntimeError("BraveSearch client has no supported search method.")

    async def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_at
        wait_for = self._min_interval - elapsed
        if wait_for > 0:
            await asyncio.sleep(wait_for)
        self._last_request_at = time.monotonic()


def _is_rate_limit_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) if response is not None else None
    if status == 429:
        return True
    if response is not None:
        status = getattr(response, "status", None)
        if status == 429:
            return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    status = getattr(exc, "status", None)
    if status == 429:
        return True
    status = getattr(exc, "code", None)
    if status == 429:
        return True
    message = str(exc)
    return "429" in message or "Too Many Requests" in message


async def _to_search_results(kernel_results) -> List[SearchResult]:
    items: List[SearchResult] = []
    if kernel_results is None:
        return items
    results_iter = _extract_iterable_results(kernel_results)
    if hasattr(results_iter, "__aiter__"):
        async for result in results_iter:
            items.append(_normalize_text_result(result))
    else:
        for result in results_iter:
            items.append(_normalize_text_result(result))
    return items


def _extract_iterable_results(kernel_results: Any):
    if hasattr(kernel_results, "results"):
        return getattr(kernel_results, "results")
    if isinstance(kernel_results, list):
        return kernel_results
    if hasattr(kernel_results, "__iter__"):
        return kernel_results
    return []


def _normalize_text_result(result) -> SearchResult:
    if isinstance(result, dict):
        name = result.get("name") or result.get("title")
        value = result.get("value") or result.get("snippet") or ""
        link = result.get("link") or result.get("url") or ""
    else:
        name = getattr(result, "name", None) or getattr(result, "title", None)
        value = getattr(result, "value", None) or getattr(result, "snippet", None) or ""
        link = getattr(result, "link", None) or getattr(result, "url", None) or ""
    value = _clean_snippet(value)
    title = name or (value[:60] + "...") if value else (link or "Result")
    return SearchResult(title=title, snippet=value or "", url=link or "")


_TAG_RE = re.compile(r"<[^>]+>")


def _clean_snippet(text: str) -> str:
    if not text:
        return ""
    return _TAG_RE.sub("", html.unescape(text)).strip()
