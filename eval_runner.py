from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

from search_plugin import SearchPlugin
from eval_logging import RunLogger
from langgraph_flow import build_graph, build_openai_client
from otel_setup import init_tracing
from search_providers import BraveSearchProvider, NoopSearchProvider


@dataclass(frozen=True)
class QueryItem:
    id: Optional[int]
    query: str
    bucket: Optional[str]
    browse_required: Optional[bool]
    grounding_required: Optional[bool]
    notes: Optional[str]


def _utc_now_slug() -> str:
    """Return a UTC timestamp slug for eval ids."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_provider(mode: str, brave_key: str | None):
    """Construct the search provider for the eval run."""
    if mode == "brave":
        return BraveSearchProvider(api_key=brave_key)
    return NoopSearchProvider()


def _ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory for a path exists."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _append_record(path: str, record: Dict[str, Any]) -> None:
    """Append a JSONL record to the eval log."""
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _shorten(text: str, limit: int = 90) -> str:
    """Shorten a string for compact logging."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _parse_int_set(value: Optional[str]) -> set[int]:
    """Parse a comma-separated list of ints into a set."""
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
    """Parse a comma-separated list of strings into a set."""
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _load_queryset(path: str) -> List[QueryItem]:
    """Load and validate a queryset JSON file."""
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
    """Filter queryset items by id and bucket."""
    filtered = list(items)
    if ids:
        filtered = [item for item in filtered if item.id in ids]
    if buckets:
        filtered = [item for item in filtered if item.bucket in buckets]
    return filtered


def _load_completed_queries(log_path: str, eval_id: str) -> set[str]:
    """Load completed queries for a given eval id from JSONL logs."""
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
    """Run a single query through the LangGraph pipeline and log results."""
    run_logger = RunLogger(eval_id=eval_id, log_path=eval_log)
    run_logger.record_query(item.query)
    run_logger.set_model_id(model)
    search_plugin = SearchPlugin(provider, run_logger=run_logger)
    app = build_graph(
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
    """CLI entrypoint for running evals over a queryset."""
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
    parser.add_argument("--search", default="brave", choices=["none", "brave"])
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
    provider = _build_provider(args.search, brave_api_key)
    client, model = build_openai_client()

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
