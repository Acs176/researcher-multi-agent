from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from bus import Message, MessageBus
from prompts import CRITIC_SYSTEM, PLANNER_SYSTEM, SUMMARIZER_SYSTEM, WRITER_SYSTEM
from search_providers import SearchProvider, SearchResult
from sk_client import SKClient
from utils import extract_json, require_json, to_json_text


class BaseAgent:
    def __init__(self, name: str, bus: MessageBus, llm: SKClient) -> None:
        self.name = name
        self.bus = bus
        self.llm = llm
        self.logger = logging.getLogger(f"agent.{name.lower()}")

    async def send(
        self,
        msg_type: str,
        content: str,
        citations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        in_reply_to: Optional[str] = None,
    ) -> None:
        self.logger.info("send type=%s reply_to=%s", msg_type, in_reply_to)
        message = self.bus.new_message(
            sender=self.name,
            msg_type=msg_type,
            content=content,
            citations=citations,
            metadata=metadata,
            in_reply_to=in_reply_to,
        )
        await self.bus.publish(message)

    def _log_llm_output(self, label: str, text: str) -> None:
        preview = _truncate(text, 500)
        self.logger.info("%s output=%s", label, preview)



class PlannerAgent(BaseAgent):
    def register(self) -> None:
        self.bus.subscribe("user_query", self.handle)

    async def handle(self, message: Message) -> None:
        self.logger.info("recv type=%s from=%s", message.type, message.sender)
        result = await self.llm.complete(
            PLANNER_SYSTEM,
            message.content,
            response_format={"type": "json_object"},
        )
        self._log_llm_output("planner", result)
        try:
            plan_json = require_json(result)
        except ValueError:
            self.logger.error("planner invalid json: %s", _truncate(result, 1000))
            raise
        plan_json = _normalize_plan(plan_json)
        if plan_json.get("needs_clarification"):
            clarification = plan_json.get("clarification") or {}
            question = clarification.get("question") or "Could you clarify your request?"
            await self.send(
                "clarification_request",
                question,
                metadata=plan_json,
                in_reply_to=message.id,
            )
            return
        payload = f"{result}\n\nJSON:\n{to_json_text(plan_json)}"
        await self.send(
            "plan",
            payload,
            metadata={"plan": plan_json, "user_query": message.content},
            in_reply_to=message.id,
        )


class SearchAgent(BaseAgent):
    def __init__(self, name: str, bus: MessageBus, llm: SKClient, provider: SearchProvider) -> None:
        super().__init__(name, bus, llm)
        self.provider = provider
        self._results: List[SearchResult] = []

    def register(self) -> None:
        self.bus.subscribe("plan", self.handle_plan)
        self.bus.subscribe("info_request", self.handle_request)

    async def handle_plan(self, message: Message) -> None:
        self.logger.info("recv plan from=%s", message.sender)
        plan = (message.metadata or {}).get("plan", {})
        if plan.get("needs_clarification"):
            self.logger.info("skip search due to clarification request")
            return
        user_query = (message.metadata or {}).get("user_query", "")
        queries = plan.get("search_queries") or [user_query or message.content]
        await self._run_searches(queries, in_reply_to=message.id, user_query=user_query)

    async def handle_request(self, message: Message) -> None:
        self.logger.info("recv info_request from=%s", message.sender)
        req = extract_json(message.content) or {}
        intent = (req.get("intent") or "").lower()
        if intent != "search":
            self.logger.info("ignore info_request intent=%s", intent or "none")
            return
        query = req.get("query")
        if not query:
            self.logger.info("ignore info_request with empty query")
            return
        await self._run_searches([query], in_reply_to=message.id, user_query="")

    async def _run_searches(
        self,
        queries: List[str],
        in_reply_to: Optional[str],
        user_query: str,
    ) -> None:
        for query in queries:
            if not query:
                continue
            if "{$input}" in query:
                query = query.replace("{$input}", user_query or "").strip()
            if not query:
                self.logger.info("skip empty query after substitution")
                continue
            self.logger.info("search query=%s", query)
            try:
                results = await self.provider.search(query)
            except Exception as exc:  # pragma: no cover - external dependency
                self.logger.warning("search failed query=%s error=%s", query, exc, exc_info=True)
                continue
            self._results.extend(results)
        self.logger.info("search results total=%s", len(self._results))
        formatted = _format_results(self._results)
        metadata = {"results": [r.__dict__ for r in self._results]}
        await self.send("search_results", formatted, metadata=metadata, in_reply_to=in_reply_to)


class SummarizerAgent(BaseAgent):
    def register(self) -> None:
        self.bus.subscribe("search_results", self.handle)

    async def handle(self, message: Message) -> None:
        self.logger.info("recv search_results from=%s", message.sender)
        results = (message.metadata or {}).get("results", [])
        user_prompt = _results_to_prompt(results)
        result = await self.llm.complete(
            SUMMARIZER_SYSTEM,
            user_prompt,
            response_format={"type": "json_object"},
        )
        self._log_llm_output("summarizer", result)
        try:
            summary_json = require_json(result)
        except ValueError:
            self.logger.error("summarizer invalid json: %s", _truncate(result, 1000))
            raise
        payload = to_json_text(summary_json)
        await self.send("summary", payload, metadata=summary_json, in_reply_to=message.id)


class CriticAgent(BaseAgent):
    def __init__(self, name: str, bus: MessageBus, llm: SKClient, max_rounds: int = 2) -> None:
        super().__init__(name, bus, llm)
        self.max_rounds = max_rounds
        self._rounds = 0

    def register(self) -> None:
        self.bus.subscribe("summary", self.handle)

    async def handle(self, message: Message) -> None:
        self.logger.info("recv summary from=%s", message.sender)
        if self._rounds >= self.max_rounds:
            await self.send(
                "critique",
                to_json_text({"status": "ok", "issues": [], "requests": []}),
                metadata={"status": "ok"},
                in_reply_to=message.id,
            )
            return
        result = await self.llm.complete(
            CRITIC_SYSTEM,
            message.content,
            response_format={"type": "json_object"},
        )
        self._log_llm_output("critic", result)
        try:
            critique = require_json(result)
        except ValueError:
            self.logger.error("critic invalid json: %s", _truncate(result, 1000))
            raise
        self._rounds += 1
        await self.send("critique", to_json_text(critique), metadata=critique, in_reply_to=message.id)
        if critique.get("status") == "needs_more":
            for req in critique.get("requests", []):
                await self.send("info_request", to_json_text(req), in_reply_to=message.id)


class WriterAgent(BaseAgent):
    def __init__(self, name: str, bus: MessageBus, llm: SKClient) -> None:
        super().__init__(name, bus, llm)
        self._latest_summary: Optional[str] = None

    def register(self) -> None:
        self.bus.subscribe("summary", self.handle_summary)
        self.bus.subscribe("critique", self.handle_critique)

    async def handle_summary(self, message: Message) -> None:
        self.logger.info("recv summary from=%s", message.sender)
        self._latest_summary = message.content

    async def handle_critique(self, message: Message) -> None:
        self.logger.info("recv critique from=%s", message.sender)
        critique = require_json(message.content)
        if critique.get("status") != "ok":
            return
        summary = self._latest_summary or message.content
        result = await self.llm.complete(WRITER_SYSTEM, summary)
        self._log_llm_output("writer", result)
        await self.send("final_answer", result, in_reply_to=message.id)


def _format_results(results: List[SearchResult]) -> str:
    lines: List[str] = []
    for idx, res in enumerate(results, start=1):
        lines.append(f"[{idx}] {res.title} - {res.url}")
        if res.snippet:
            lines.append(res.snippet)
        lines.append("")
    return "\n".join(lines).strip()


def _results_to_prompt(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, res in enumerate(results, start=1):
        title = res.get("title", "Untitled")
        url = res.get("url", "")
        snippet = res.get("snippet", "")
        lines.append(f"[{idx}] {title} | {url}")
        if snippet:
            lines.append(snippet)
        lines.append("")
    return "\n".join(lines).strip() if lines else "No results."


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _normalize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    logger = logging.getLogger("agent.planner")
    allowed_agents = {"Planner", "Search", "Summarizer", "Critic", "Writer"}
    normalized = dict(plan)
    order = normalized.get("order")
    if isinstance(order, list):
        unknown_in_order = [agent for agent in order if agent not in allowed_agents]
        if unknown_in_order:
            logger.warning("planner used unknown agents in order: %s", unknown_in_order)
        normalized["order"] = [agent for agent in order if agent in allowed_agents]
    tasks = normalized.get("tasks")
    if isinstance(tasks, list):
        cleaned: List[Dict[str, Any]] = []
        unknown_task_agents: List[str] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_copy = dict(task)
            agent = task_copy.get("agent")
            if agent not in allowed_agents:
                if agent:
                    unknown_task_agents.append(agent)
                task_copy.pop("agent", None)
            cleaned.append(task_copy)
        if unknown_task_agents:
            logger.warning("planner used unknown agents in tasks: %s", unknown_task_agents)
        normalized["tasks"] = cleaned
    return normalized
