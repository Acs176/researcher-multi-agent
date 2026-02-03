from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from agents import SearchPlugin
from eval_logging import RunLogger
from otel_setup import init_tracing
from polite_fetcher import PoliteFetcher
from prompts import SEARCH_PLAN_SYSTEM, SEARCH_URLS_SYSTEM, SUMMARY_SYSTEM
from search_providers import (
    LocalDocSearchProvider,
    NoopSearchProvider,
    SemanticKernelBraveSearchProvider,
)
from utils import extract_json

import ftfy


@dataclass(frozen=True)
class QueryItem:
    id: Optional[int]
    query: str
    bucket: Optional[str]
    browse_required: Optional[bool]
    grounding_required: Optional[bool]
    notes: Optional[str]


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
        if not self.needs_search:
            self.queries = []
            self.urls = []
        self.queries = [q.strip() for q in self.queries if isinstance(q, str) and q.strip()]
        self.urls = [u.strip() for u in self.urls if isinstance(u, str) and u.strip()]
        return self


def _utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_provider(mode: str, docs_root: str, brave_key: str | None):
    if mode == "local":
        return LocalDocSearchProvider(docs_root)
    if mode == "brave":
        return SemanticKernelBraveSearchProvider(api_key=brave_key)
    return NoopSearchProvider()


def _build_openai_client() -> tuple[Any, str]:
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


def _build_fetcher() -> PoliteFetcher:
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


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _append_record(path: str, record: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

async def _call_model(
    client: Any,
    model: str,
    system_prompt: str,
    user_text: str,
    schema_model: type[BaseModel] | None,
    timeout: float,
    run_logger: RunLogger,
) -> Any:
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
        except Exception as e:
            print(
                f"EXCEPTION IN CHAT COMPLETION {e} - Retrying with json_object response format"
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


def _shorten(text: str, limit: int = 90) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _clean_text(text: str) -> str:
    cleaned = text
    if ftfy is not None:
        cleaned = ftfy.fix_text(cleaned)
    # Strip control chars while keeping newlines and tabs.
    cleaned = "".join(ch for ch in cleaned if ch >= " " or ch in "\n\t")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _build_summary_input(question: str, fetched: Sequence[Dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(fetched, start=1):
        if not isinstance(item, dict):
            continue
        text = item.get("text") or ""
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


def _parse_int_set(value: Optional[str]) -> set[int]:
    if not value:
        return set()
    items: set[int] = set()
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        items.add(int(raw))
    return items


def _parse_str_set(value: Optional[str]) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _load_queryset(path: str) -> List[QueryItem]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Queryset must be a JSON array.")
    items: List[QueryItem] = []
    for idx, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise ValueError(f"Queryset item {idx} must be an object.")
        query = str(raw.get("query") or "").strip()
        if not query:
            raise ValueError(f"Queryset item {idx} missing query.")
        items.append(
            QueryItem(
                id=raw.get("id"),
                query=query,
                bucket=raw.get("bucket"),
                browse_required=raw.get("browse_required"),
                grounding_required=raw.get("grounding_required"),
                notes=raw.get("notes"),
            )
        )
    return items


def _filter_items(
    items: Sequence[QueryItem],
    ids: set[int],
    buckets: set[str],
) -> List[QueryItem]:
    filtered = list(items)
    if ids:
        filtered = [item for item in filtered if item.id in ids]
    if buckets:
        filtered = [item for item in filtered if item.bucket in buckets]
    return filtered


def _load_completed_queries(log_path: str, eval_id: str) -> set[str]:
    if not os.path.exists(log_path):
        return set()
    completed: set[str] = set()
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("eval_id") != eval_id:
                continue
            query = record.get("query")
            if isinstance(query, str) and query:
                completed.add(query)
    return completed


def _build_graph(
    client: Any,
    model: str,
    search_plugin: SearchPlugin,
    run_logger: RunLogger,
    timeout: float,
    show_steps: bool,
) -> Any:
    async def plan_node(state: AgentState) -> Dict[str, Any]:
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
            print("\n--- URLs Node Output ---")
            print(json.dumps({"urls": selection.urls, "plan": selection.model_dump()}, ensure_ascii=True))
        return {"urls": selection.urls, "plan": selection.model_dump()}

    async def fetch_node(state: AgentState) -> Dict[str, Any]:
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
                    "text": _clean_text(res.text),
                    "robots_allowed": res.robots_allowed,
                    "error": res.error,
                }
            )
        if show_steps:
            print("\n--- Fetch Output ---")
            print(json.dumps(fetched, ensure_ascii=True))
        return {"fetched": fetched}

    async def summarize_node(state: AgentState) -> Dict[str, Any]:
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
        plan = SearchPlan.model_validate(state["plan"])
        if not plan.needs_search or not plan.queries:
            return "summarize"
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
            "summarize": "summarize",
        },
    )
    graph.add_edge("search", "select_urls")
    graph.add_edge("select_urls", "fetch")
    graph.add_edge("fetch", "summarize")
    graph.add_edge("summarize", END)
    graph.set_entry_point("planning")
    return graph.compile()


async def _run_item(
    item: QueryItem,
    client: Any,
    model: str,
    provider,
    eval_id: str,
    eval_log: str,
    timeout: float,
    show_steps: bool,
) -> None:
    run_logger = RunLogger(eval_id=eval_id, log_path=eval_log)
    run_logger.record_query(item.query)
    run_logger.set_model_id(model)
    search_plugin = SearchPlugin(provider, run_logger=run_logger)
    app = _build_graph(
        client=client,
        model=model,
        search_plugin=search_plugin,
        run_logger=run_logger,
        timeout=timeout,
        show_steps=show_steps,
    )

    started_at = time.monotonic()
    try:
        result = await app.ainvoke(
            {
                "query": item.query,
                "plan": {},
                "search_results": [],
                "urls": [],
                "fetched": [],
                "answer": "",
            }
        )
        answer = (result.get("answer") or "").strip()
        if not answer:
            fetched = result.get("fetched", [])
            answer = json.dumps(fetched, ensure_ascii=True)
        run_logger.record_answer(answer)
        run_logger.record_latency(started_at)
        run_logger.write()
    except Exception as exc:
        run_logger.record_latency(started_at)
        record = run_logger.to_record()
        record["error"] = f"{type(exc).__name__}: {exc}"
        _append_record(eval_log, record)
        raise


async def run() -> None:
    parser = argparse.ArgumentParser(description="LangGraph eval runner")
    parser.add_argument("--queryset", default="eval/queryset.json", help="Path to queryset JSON")
    parser.add_argument(
        "--eval-id",
        default=_utc_now_slug(),
        help="Eval id for grouping runs (default: timestamp UTC)",
    )
    parser.add_argument(
        "--eval-log",
        default="eval/eval_runs.jsonl",
        help="Path to append eval run logs (JSONL)",
    )
    parser.add_argument("--docs-root", default=os.getcwd(), help="Root for local doc search")
    parser.add_argument("--search", default="brave", choices=["local", "none", "brave"])
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    parser.add_argument("--show-steps", action="store_true", help="Print agent outputs as they complete")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of queries to run")
    parser.add_argument("--ids", help="Comma-separated query ids to run")
    parser.add_argument("--bucket", help="Comma-separated bucket names to run")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle query order")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed")
    parser.add_argument("--resume", action="store_true", help="Skip queries already logged")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    args = parser.parse_args()

    load_dotenv()
    init_tracing()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    items = _load_queryset(args.queryset)
    ids = _parse_int_set(args.ids)
    buckets = _parse_str_set(args.bucket)
    items = _filter_items(items, ids=ids, buckets=buckets)
    if args.shuffle:
        rng = random.Random(args.seed)
        items = list(items)
        rng.shuffle(items)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    if not items:
        print("No queries to run after filtering.")
        return

    completed = set()
    if args.resume:
        completed = _load_completed_queries(args.eval_log, args.eval_id)

    brave_api_key = os.getenv("BRAVE_API_KEY")
    provider = _build_provider(args.search, args.docs_root, brave_api_key)
    client, model = _build_openai_client()

    total = len(items)
    failures = 0
    ran = 0
    for idx, item in enumerate(items, start=1):
        if args.resume and item.query in completed:
            continue
        prefix = f"[{idx}/{total}]"
        item_id = f"id={item.id}" if item.id is not None else "id=?"
        bucket = f"bucket={item.bucket}" if item.bucket else "bucket=?"
        print(f"{prefix} {item_id} {bucket} query={_shorten(item.query)}")
        try:
            await _run_item(
                item=item,
                client=client,
                model=model,
                provider=provider,
                eval_id=args.eval_id,
                eval_log=args.eval_log,
                timeout=args.timeout,
                show_steps=args.show_steps,
            )
            ran += 1
        except Exception as exc:
            failures += 1
            logging.exception("Query failed: %s", exc)
            if args.fail_fast:
                break

    print(
        "Eval complete: eval_id=%s ran=%s failed=%s log=%s"
        % (args.eval_id, ran, failures, args.eval_log)
    )


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        sys.exit(130)
