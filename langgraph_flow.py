from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph import END, StateGraph
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from eval_logging import RunLogger
from polite_fetcher import PoliteFetcher
from prompts import SEARCH_PLAN_SYSTEM, SEARCH_URLS_SYSTEM, SUMMARY_SYSTEM
from utils import extract_json

import ftfy


class AgentState(TypedDict):
    query: str
    plan: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    urls: List[str]
    fetched: List[Dict[str, Any]]
    answer: str


class SearchPlan(BaseModel):
    needs_search: bool
    queries: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)
    reasoning: str

    @model_validator(mode="after")
    def _normalize(self) -> "SearchPlan":
        """Normalize query/url lists after validation."""
        if not self.needs_search:
            self.queries = []
            self.urls = []
        self.queries = [q.strip() for q in self.queries if isinstance(q, str) and q.strip()]
        self.urls = [u.strip() for u in self.urls if isinstance(u, str) and u.strip()]
        return self


def build_openai_client() -> tuple[Any, str]:
    """Construct an OpenAI async client and model identifier."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    org_id = os.getenv("OPENAI_ORG_ID")
    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if org_id:
        client_kwargs["organization"] = org_id
    client = AsyncOpenAI(**client_kwargs)
    model = os.getenv("SK_MODEL", "gpt-5-mini")
    return client, model


def build_graph(
    client: Any,
    model: str,
    search_plugin: Any,
    run_logger: RunLogger,
    timeout: float,
    show_steps: bool,
) -> Any:
    """Build the LangGraph pipeline for planning, searching, fetching, and summarizing."""

    async def plan_node(state: AgentState) -> Dict[str, Any]:
        """Decide whether to search and propose queries."""
        plan = await _call_model(
            client=client,
            model=model,
            system_prompt=SEARCH_PLAN_SYSTEM,
            user_text=state["query"],
            schema_model=SearchPlan,
            timeout=timeout,
            run_logger=run_logger,
        )
        if show_steps:
            print("\n--- Plan Output ---")
            print(json.dumps(plan.model_dump(), ensure_ascii=True))
        return {
            "plan": plan.model_dump(),
            "search_results": [],
            "urls": [],
            "fetched": [],
            "answer": "",
        }

    async def search_node(state: AgentState) -> Dict[str, Any]:
        """Execute web searches and collect normalized results."""
        plan = SearchPlan.model_validate(state["plan"])
        if not plan.needs_search or not plan.queries:
            return {"search_results": []}
        results: List[Dict[str, Any]] = []
        for query in plan.queries:
            payload_text = await search_plugin.search(query)
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                print("Exception decoding payload from search result")
                continue
            for item in payload.get("results", []):
                if not isinstance(item, dict):
                    continue
                results.append(
                    {
                        "title": item.get("title") or "",
                        "snippet": item.get("snippet") or "",
                        "url": item.get("url") or "",
                        "query": item.get("query") or query,
                    }
                )
        if show_steps:
            print("\n--- Search Output ---")
            print(json.dumps(results, ensure_ascii=True))
        return {"search_results": results}

    async def urls_node(state: AgentState) -> Dict[str, Any]:
        """Select URLs to fetch based on search results."""
        plan = SearchPlan.model_validate(state["plan"])
        if not plan.needs_search:
            return {"urls": []}
        user_payload = {
            "question": state["query"],
            "queries": plan.queries,
            "results": state["search_results"],
        }
        selection = await _call_model(
            client=client,
            model=model,
            system_prompt=SEARCH_URLS_SYSTEM,
            user_text=json.dumps(user_payload, ensure_ascii=True),
            schema_model=SearchPlan,
            timeout=timeout,
            run_logger=run_logger,
        )
        if show_steps:
            print("\n--- URL picker Output ---")
            print(json.dumps(selection.urls, ensure_ascii=True))
        return {"urls": selection.urls, "plan": selection.model_dump()}

    async def fetch_node(state: AgentState) -> Dict[str, Any]:
        """Fetch the selected URLs and return extracted text."""
        urls = state.get("urls") or []
        if not urls:
            return {"fetched": []}
        fetcher = _build_fetcher()
        try:
            fetched_map = await fetcher.fetch_many(urls)
        finally:
            await fetcher.close()
        fetched: List[Dict[str, Any]] = []
        for url in urls:
            res = fetched_map.get(url)
            if not res:
                continue
            fetched.append(
                {
                    "url": res.url,
                    "status": res.status,
                    "text": res.text,
                    "robots_allowed": res.robots_allowed,
                    "error": res.error,
                }
            )
        if show_steps:
            print("\n--- Fetch Output ---")
            print(json.dumps(fetched, ensure_ascii=True))
        return {"fetched": fetched}

    async def summarize_node(state: AgentState) -> Dict[str, Any]:
        """Summarize the fetched content into a final answer."""
        user_text = _build_summary_input(state["query"], state.get("fetched") or [])
        answer = await _call_model(
            client=client,
            model=model,
            system_prompt=SUMMARY_SYSTEM,
            user_text=user_text,
            schema_model=None,
            timeout=timeout,
            run_logger=run_logger,
        )
        if show_steps:
            print("\n--- Summary Output ---")
            print(answer)
        return {"answer": answer}

    def plan_route(state: AgentState) -> str:
        """Route to search or fetch based on the plan decision."""
        plan = SearchPlan.model_validate(state["plan"])
        if not plan.needs_search or not plan.queries:
            return "fetch"
        return "search"

    graph = StateGraph(AgentState)
    graph.add_node("planning", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("select_urls", urls_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("summarize", summarize_node)
    graph.add_conditional_edges(
        "planning",
        plan_route,
        {
            "search": "search",
            "fetch": "fetch",
        },
    )
    graph.add_edge("search", "select_urls")
    graph.add_edge("select_urls", "fetch")
    graph.add_edge("fetch", "summarize")
    graph.add_edge("summarize", END)
    graph.set_entry_point("planning")
    return graph.compile()


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


def _build_fetcher() -> PoliteFetcher:
    """Create a rate-limited fetcher for URL content."""
    return PoliteFetcher(
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


def _clean_text(text: str) -> str:
    """Lightly normalize extracted text and remove control characters."""
    cleaned = ftfy.fix_text(text)
    cleaned = "".join(ch for ch in cleaned if ch >= " " or ch in "\n\t")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _build_summary_input(question: str, fetched: Sequence[Dict[str, Any]]) -> str:
    """Build the summarizer input from fetched sources."""
    blocks: list[str] = []
    for idx, item in enumerate(fetched, start=1):
        if not isinstance(item, dict):
            continue
        text = _clean_text(item.get("text") or "")
        if not text:
            continue
        url = item.get("url") or ""
        status = item.get("status")
        block = [
            f"[Source {idx}]",
            f"URL: {url}",
            f"Status: {status}",
            "Text:",
            text,
        ]
        blocks.append("\n".join(block))
    if not blocks:
        content = "No fetched content was available."
    else:
        content = "\n\n".join(blocks)
    return "\n\n".join(
        [
            f"Question: {question}",
            "Fetched content:",
            content,
        ]
    )


async def _call_model(
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    schema_model: type[BaseModel] | None,
    timeout: float,
    run_logger: RunLogger,
) -> Any:
    """Call the model and optionally parse structured output."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    if schema_model is not None:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=schema_model,
                ),
                timeout=timeout,
            )
        except Exception as exc:
            print(
                "EXCEPTION IN CHAT COMPLETION "
                f"{exc} - Retrying with json_object response format"
            )
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                ),
                timeout=timeout,
            )
    else:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=messages,
            ),
            timeout=timeout,
        )
    run_logger.record_usage(getattr(response, "usage", None))
    run_logger.set_model_id(getattr(response, "model", None))
    content = response.choices[0].message.content or ""
    if schema_model is None:
        return content.strip()
    try:
        return schema_model.model_validate_json(content)
    except ValidationError:
        recovered = extract_json(content)
        if recovered is None:
            raise
        return schema_model.model_validate(recovered)
