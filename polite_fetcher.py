from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from opentelemetry import trace
import trafilatura  # type: ignore
import redis.asyncio as redis  # type: ignore



@dataclass
class FetchResult:
    url: str
    status: int
    text: str
    robots_allowed: bool
    error: Optional[str] = None


@dataclass
class RobotsEntry:
    parser: RobotFileParser
    fetched_at: float
    deny_all: bool


class RobotsDenyCache:
    """Redis-backed cache for domains that deny all robots."""

    def __init__(self, redis_url: str, ttl_seconds: int, key_prefix: str = "robots:deny") -> None:
        """Initialize a deny cache with a Redis connection URL and TTL."""
        self._redis_url = redis_url
        self._ttl_seconds = max(0, ttl_seconds)
        self._key_prefix = key_prefix.rstrip(":")
        self._redis: "redis.Redis" = redis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        self._logger = logging.getLogger("fetcher.robots")

    async def is_denied(self, scheme: str, netloc: str) -> bool:
        """Return True if the domain is cached as deny-all."""
        key = self._key(scheme, netloc)
        try:
            return await self._redis.exists(key) == 1
        except Exception as exc:  # pragma: no cover - redis unavailable
            self._logger.info("robots deny cache check failed key=%s error=%s", key, exc)
            return False

    async def mark_denied(self, scheme: str, netloc: str) -> None:
        """Mark the domain as deny-all with the configured TTL."""
        if self._ttl_seconds <= 0:
            return
        key = self._key(scheme, netloc)
        try:
            await self._redis.set(key, "1", ex=self._ttl_seconds)
        except Exception as exc:  # pragma: no cover - redis unavailable
            self._logger.info("robots deny cache set failed key=%s error=%s", key, exc)

    async def close(self) -> None:
        """Close the Redis connection if initialized."""
        try:
            close_result = self._redis.close()
            if asyncio.iscoroutine(close_result):
                await close_result
        except Exception as exc:  # pragma: no cover - redis unavailable
            self._logger.info("robots deny cache close failed error=%s", exc)

    def _key(self, scheme: str, netloc: str) -> str:
        return f"{self._key_prefix}:{scheme}://{netloc}"


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
        deny_cache: Optional[RobotsDenyCache] = None,
    ) -> None:
        self._client = client
        self._limiter = limiter
        self._user_agent = user_agent
        self._ttl_seconds = max(0, ttl_seconds)
        self._cache: Dict[str, RobotsEntry] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._deny_cache = deny_cache
        self._logger = logging.getLogger("fetcher.robots")

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return True
        # scheme: protocol (http/https), netloc: host[:port] (authority portion of the URL).
        if self._deny_cache and await self._deny_cache.is_denied(parsed.scheme, parsed.netloc):
            return False
        entry = await self._get_entry(parsed.scheme, parsed.netloc)
        if entry.deny_all:
            if self._deny_cache:
                await self._deny_cache.mark_denied(parsed.scheme, parsed.netloc)
            return False
        return entry.parser.can_fetch(self._user_agent, url)

    async def _get_entry(self, scheme: str, netloc: str) -> RobotsEntry:
        cache_key = f"{scheme}://{netloc}"
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached.fetched_at) < self._ttl_seconds:
            return cached
        lock = self._locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            cached = self._cache.get(cache_key)
            if cached and (time.time() - cached.fetched_at) < self._ttl_seconds:
                return cached
            entry = await self._fetch_entry(scheme, netloc)
            self._cache[cache_key] = entry
            return entry

    async def _fetch_entry(self, scheme: str, netloc: str) -> RobotsEntry:
        robots_url = f"{scheme}://{netloc}/robots.txt"
        await self._limiter.wait(netloc)
        try:
            resp = await self._client.get(robots_url)
        except Exception as exc:  # pragma: no cover - network error
            self._logger.info("robots fetch failed url=%s error=%s", robots_url, exc)
            return RobotsEntry(parser=_allow_all_parser(robots_url), fetched_at=time.time(), deny_all=False)
        if resp.status_code in {401, 403}:
            return RobotsEntry(parser=_deny_all_parser(robots_url), fetched_at=time.time(), deny_all=True)
        if resp.status_code >= 400:
            return RobotsEntry(parser=_allow_all_parser(robots_url), fetched_at=time.time(), deny_all=False)
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(resp.text.splitlines())
        deny_all = _robots_text_is_deny_all(resp.text)
        return RobotsEntry(parser=parser, fetched_at=time.time(), deny_all=deny_all)

    async def close(self) -> None:
        """Close any underlying deny cache resources."""
        if self._deny_cache:
            await self._deny_cache.close()


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
        robots_deny_cache: Optional[RobotsDenyCache] = None,
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
            deny_cache=robots_deny_cache,
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

    async def filter_allowed(self, urls: Iterable[str]) -> list[str]:
        """Return URLs that pass the robots allow check."""
        allowed: list[str] = []
        seen: set[str] = set()
        for url in urls:
            if not url or url in seen:
                continue
            seen.add(url)
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            if await self._robots.allowed(url):
                allowed.append(url)
        return allowed

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
        await self._robots.close()


def extract_main_text(html_text: str) -> str:
    try:
        text = trafilatura.extract(
            html_text,
            include_comments=False,
            include_tables=False,
            include_formatting=False,
        )
        if text:
            return text
    except Exception as exc:  # pragma: no cover - extraction error
        logging.getLogger("fetcher.extract").info("text extraction failed error=%s", exc)
    return ""


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


def _robots_text_is_deny_all(text: str) -> bool:
    """Return True if robots.txt denies all for User-agent: *."""
    # We only care about the group that targets User-agent: * and we treat it
    # as deny-all when it contains "Disallow: /" and has no Allow rules.
    # This parser is intentionally simple and only tracks User-agent/Allow/Disallow.
    groups: list[tuple[list[str], list[tuple[str, str]]]] = []
    current_agents: list[str] = []
    current_rules: list[tuple[str, str]] = []
    seen_rule = False

    def flush() -> None:
        nonlocal current_agents, current_rules, seen_rule
        # A group is the set of rules that follows one or more User-agent lines.
        if current_agents:
            groups.append((current_agents, current_rules))
        current_agents = []
        current_rules = []
        seen_rule = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            # Blank line closes the current group, if any.
            if current_agents or seen_rule:
                flush()
            continue
        if line.startswith("#"):
            continue
        if "#" in line:
            # Strip inline comments.
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
        if ":" not in line:
            continue
        field, value = line.split(":", 1)
        field = field.strip().lower()
        value = value.strip()
        if field == "user-agent":
            # New user-agent after rules means a new group.
            if seen_rule:
                flush()
            current_agents.append(value.lower())
        elif field in {"disallow", "allow"}:
            seen_rule = True
            current_rules.append((field, value))

    if current_agents:
        groups.append((current_agents, current_rules))

    for agents, rules in groups:
        if "*" not in agents:
            continue
        # Any Allow rule for * means we do not treat this as deny-all.
        if any(field == "allow" and value for field, value in rules):
            return False
        for field, value in rules:
            # The minimal signal for deny-all is "Disallow: /".
            if field == "disallow" and value.strip() == "/":
                return True
    return False
