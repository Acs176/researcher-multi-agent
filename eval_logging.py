from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _extract_usage_from_dict(data: Dict[str, Any]) -> Dict[str, Optional[int]]:
    usage = data.get("usage") or data.get("token_usage") or {}
    if isinstance(usage, dict):
        prompt = _coerce_int(
            usage.get("prompt_tokens")
            or usage.get("prompt")
            or usage.get("input_tokens")
        )
        completion = _coerce_int(
            usage.get("completion_tokens")
            or usage.get("completion")
            or usage.get("output_tokens")
        )
        total = _coerce_int(
            usage.get("total_tokens") or usage.get("total") or usage.get("tokens")
        )
        return {"prompt": prompt, "completion": completion, "total": total}

    prompt = _coerce_int(data.get("prompt_tokens") or data.get("prompt"))
    completion = _coerce_int(data.get("completion_tokens") or data.get("completion"))
    total = _coerce_int(data.get("total_tokens") or data.get("total"))
    return {"prompt": prompt, "completion": completion, "total": total}


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _merge_usage(
    current: Dict[str, Optional[int]], incoming: Dict[str, Optional[int]]
) -> Dict[str, Optional[int]]:
    merged = dict(current)
    for key in ("prompt", "completion", "total"):
        value = incoming.get(key)
        if value is None:
            continue
        if merged.get(key) is None:
            merged[key] = value
        else:
            merged[key] = int(merged[key]) + int(value)
    return merged


def _extract_model_id(data: Dict[str, Any]) -> Optional[str]:
    for key in ("model", "model_id", "ai_model_id", "model_name"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _iter_messages(message: Any) -> Iterable[Any]:
    if isinstance(message, list):
        return message
    return [message]


def _extract_usage_and_model(message: Any) -> tuple[Dict[str, Optional[int]], Optional[str]]:
    usage: Dict[str, Optional[int]] = {"prompt": None, "completion": None, "total": None}
    model_id: Optional[str] = None
    for item in _iter_messages(message):
        metadata = getattr(item, "metadata", None)
        if isinstance(metadata, dict):
            extracted = _extract_usage_from_dict(metadata)
            usage = _merge_usage(usage, extracted)
            model_id = model_id or _extract_model_id(metadata)
            if extracted.get("prompt") is not None or extracted.get("completion") is not None:
                continue

        inner = getattr(item, "inner_content", None)
        if isinstance(inner, dict):
            extracted = _extract_usage_from_dict(inner)
            usage = _merge_usage(usage, extracted)
            model_id = model_id or _extract_model_id(inner)
            continue

        for attr_name in ("usage", "token_usage"):
            inner_usage = getattr(item, attr_name, None)
            if isinstance(inner_usage, dict):
                usage = _merge_usage(usage, _extract_usage_from_dict(inner_usage))

        if model_id is None:
            for attr_name in ("model", "model_id", "ai_model_id", "model_name"):
                value = getattr(item, attr_name, None)
                if isinstance(value, str) and value:
                    model_id = value
                    break
    return usage, model_id


@dataclass
class RunLogger:
    eval_id: str
    log_path: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=_utc_now_iso)
    query: Optional[str] = None
    answer: Optional[str] = None
    browse_calls: int = 0
    sources: list[Dict[str, Any]] = field(default_factory=list)
    token_counts: Dict[str, Optional[int]] = field(
        default_factory=lambda: {"prompt": None, "completion": None, "total": None}
    )
    model_id: Optional[str] = None
    latency_ms: Optional[float] = None
    _seen_message_ids: set[int] = field(default_factory=set, init=False, repr=False)

    def record_query(self, query: str) -> None:
        self.query = query

    def record_search(self, query: str, results: list[Dict[str, Any]]) -> None:
        self.browse_calls += 1
        retrieved_at = _utc_now_iso()
        for item in results:
            if not isinstance(item, dict):
                continue
            snippets = []
            snippet = item.get("snippet")
            if isinstance(snippet, str) and snippet:
                snippets.append(snippet)
            self.sources.append(
                {
                    "url": item.get("url") or "",
                    "title": item.get("title") or "",
                    "retrieved_at": retrieved_at,
                    "snippets": snippets,
                    "query": query,
                }
            )

    def record_message(self, message: Any) -> None:
        for item in _iter_messages(message):
            item_id = id(item)
            if item_id in self._seen_message_ids:
                continue
            self._seen_message_ids.add(item_id)
            usage, model = _extract_usage_and_model(item)
            self.token_counts = _merge_usage(self.token_counts, usage)
            if model and not self.model_id:
                self.model_id = model

    def record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        usage_dict = _usage_to_dict(usage)
        self.token_counts = _merge_usage(self.token_counts, _extract_usage_from_dict(usage_dict))

    def record_answer(self, answer: str) -> None:
        self.answer = answer

    def record_latency(self, started_at: float, finished_at: Optional[float] = None) -> None:
        end = finished_at if finished_at is not None else time.monotonic()
        self.latency_ms = (end - started_at) * 1000.0

    def set_model_id(self, model_id: Optional[str]) -> None:
        if model_id and not self.model_id:
            self.model_id = model_id

    def to_record(self) -> Dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "query": self.query or "",
            "answer": self.answer or "",
            "browsed": self.browse_calls > 0,
            "browse_calls": self.browse_calls,
            "sources": self.sources,
            "latency_ms": self.latency_ms,
            "token_counts": self.token_counts,
            "model_id": self.model_id,
        }

    def write(self) -> None:
        _ensure_parent_dir(self.log_path)
        record = self.to_record()
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
