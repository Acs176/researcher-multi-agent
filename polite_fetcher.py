from __future__ import annotations

import asyncio
import html
import logging
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Dict, Iterable, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from opentelemetry import trace

try:  # optional, higher-quality extraction
    import trafilatura  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None


@dataclass
class FetchResult:
    url: str
    status: int
    text: str
    robots_allowed: bool
    error: Optional[str] = None


class DomainRateLimiter:
    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(0.0, min_interval)
        self._last_request_at: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def wait(self, domain: str) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            last = self._last_request_at.get(domain, 0.0)
            delay = self._min_interval - (now - last)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_request_at[domain] = time.monotonic()


class RobotsCache:
    def __init__(
        self,
        client: httpx.AsyncClient,
        limiter: DomainRateLimiter,
        user_agent: str,
        ttl_seconds: int,
    ) -> None:
        self._client = client
        self._limiter = limiter
        self._user_agent = user_agent
        self._ttl_seconds = max(0, ttl_seconds)
        self._cache: Dict[str, tuple[RobotFileParser, float]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._logger = logging.getLogger("fetcher.robots")

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return True
        parser = await self._get_parser(parsed.scheme, parsed.netloc)
        return parser.can_fetch(self._user_agent, url)

    async def _get_parser(self, scheme: str, netloc: str) -> RobotFileParser:
        cache_key = f"{scheme}://{netloc}"
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached[1]) < self._ttl_seconds:
            return cached[0]
        lock = self._locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._cache.get(cache_key)
            if cached and (time.time() - cached[1]) < self._ttl_seconds:
                return cached[0]
            parser = await self._fetch_parser(scheme, netloc)
            self._cache[cache_key] = (parser, time.time())
            return parser

    async def _fetch_parser(self, scheme: str, netloc: str) -> RobotFileParser:
        robots_url = f"{scheme}://{netloc}/robots.txt"
        await self._limiter.wait(netloc)
        try:
            resp = await self._client.get(robots_url)
        except Exception as exc:  # pragma: no cover - network error
            self._logger.info("robots fetch failed url=%s error=%s", robots_url, exc)
            return _allow_all_parser(robots_url)
        if resp.status_code in {401, 403}:
            return _deny_all_parser(robots_url)
        if resp.status_code >= 400:
            return _allow_all_parser(robots_url)
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(resp.text.splitlines())
        return parser


class PoliteFetcher:
    def __init__(
        self,
        min_interval: float = 1.0,
        max_concurrency: int = 2,
        max_retries: int = 2,
        user_agent: str = "researcher-multi-agent/1.0 (+contact@example.com)",
        max_text_chars: int = 2000,
        max_html_chars: int = 200_000,
        robots_ttl_seconds: int = 86_400,
        trace_content: bool = True,
        trace_content_max_chars: int = 0,
    ) -> None:
        self._logger = logging.getLogger("fetcher")
        self._limiter = DomainRateLimiter(min_interval=min_interval)
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        self._max_retries = max(0, max_retries)
        self._max_text_chars = max(0, max_text_chars)
        self._max_html_chars = max(0, max_html_chars)
        self._trace_content = trace_content
        self._trace_content_max_chars = max(0, trace_content_max_chars)
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            timeout=15.0,
            follow_redirects=True,
        )
        self._robots = RobotsCache(
            client=self._client,
            limiter=self._limiter,
            user_agent=user_agent,
            ttl_seconds=robots_ttl_seconds,
        )
        self._cache: Dict[str, FetchResult] = {}

    async def fetch_many(self, urls: Iterable[str]) -> Dict[str, FetchResult]:
        unique: Dict[str, None] = {}
        for url in urls:
            if url:
                unique[url] = None
        tasks = [asyncio.create_task(self.fetch(url)) for url in unique]
        results = await asyncio.gather(*tasks)
        return {res.url: res for res in results if res is not None}

    async def fetch(self, url: str) -> Optional[FetchResult]:
        cached = self._cache.get(url)
        if cached:
            return cached
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return None
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("fetch.page") as span:
            span.set_attribute("fetch.url", url)
            span.set_attribute("fetch.domain", parsed.netloc)
            allowed = await self._robots.allowed(url)
            span.set_attribute("fetch.robots_allowed", allowed)
            if not allowed:
                span.add_event("robots.disallowed")
                result = FetchResult(url=url, status=0, text="", robots_allowed=False)
                self._cache[url] = result
                return result
            async with self._sem:
                for attempt in range(self._max_retries + 1):
                    await self._limiter.wait(parsed.netloc)
                    try:
                        resp = await self._client.get(url)
                    except Exception as exc:  # pragma: no cover - network error
                        if attempt < self._max_retries:
                            await asyncio.sleep(0.8 * (attempt + 1))
                            continue
                        span.set_attribute("fetch.error", str(exc))
                        result = FetchResult(
                            url=url,
                            status=0,
                            text="",
                            robots_allowed=True,
                            error=str(exc),
                        )
                        self._cache[url] = result
                        return result
                    span.set_attribute("http.status_code", resp.status_code)
                    if resp.status_code in {429, 503} and attempt < self._max_retries:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            await asyncio.sleep(int(retry_after))
                        else:
                            await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    html_text = resp.text or ""
                    span.set_attribute("fetch.html_length", len(html_text))
                    if self._max_html_chars and len(html_text) > self._max_html_chars:
                        html_text = html_text[: self._max_html_chars]
                        span.set_attribute("fetch.html_truncated", True)
                    else:
                        span.set_attribute("fetch.html_truncated", False)
                    text_full = extract_main_text(html_text)
                    span.set_attribute("fetch.text_length", len(text_full))
                    if self._trace_content:
                        text_for_trace = text_full
                        truncated = False
                        if self._trace_content_max_chars and len(text_for_trace) > self._trace_content_max_chars:
                            text_for_trace = text_for_trace[: self._trace_content_max_chars] + "..."
                            truncated = True
                        span.add_event(
                            "fetch.extracted_text",
                            {"text": text_for_trace, "truncated": truncated},
                        )
                    text_for_result = text_full
                    if self._max_text_chars and len(text_for_result) > self._max_text_chars:
                        text_for_result = text_for_result[: self._max_text_chars] + "..."
                        span.set_attribute("fetch.text_truncated", True)
                    else:
                        span.set_attribute("fetch.text_truncated", False)
                    result = FetchResult(
                        url=url,
                        status=resp.status_code,
                        text=text_for_result,
                        robots_allowed=True,
                    )
                    self._cache[url] = result
                    return result
        return None

    async def close(self) -> None:
        await self._client.aclose()


def extract_main_text(html_text: str) -> str:
    if trafilatura is not None:
        try:
            text = trafilatura.extract(
                html_text,
                include_comments=False,
                include_tables=False,
                include_formatting=False,
            )
            if text:
                return text
        except Exception:  # pragma: no cover - optional dependency
            pass
    return _strip_html(html_text)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        text = " ".join(part.strip() for part in self._parts if part.strip())
        return html.unescape(text)


def _strip_html(html_text: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html_text)
    return parser.get_text()


def _allow_all_parser(robots_url: str) -> RobotFileParser:
    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse([])
    return parser


def _deny_all_parser(robots_url: str) -> RobotFileParser:
    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.parse(["User-agent: *", "Disallow: /"])
    return parser
