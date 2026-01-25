from __future__ import annotations

import asyncio
import logging
from typing import Optional

from agents import CriticAgent, PlannerAgent, SearchAgent, SummarizerAgent, WriterAgent
from bus import Message, MessageBus
from search_providers import SearchProvider
from sk_client import SKClient


class ResearchCoordinator:
    def __init__(
        self,
        llm: SKClient,
        search_provider: SearchProvider,
        max_rounds: int = 2,
    ) -> None:
        self.bus = MessageBus()
        self.llm = llm
        self.search_provider = search_provider
        self.max_rounds = max_rounds
        self._final_future: Optional[asyncio.Future[str]] = None
        self._register_agents()
        self.bus.subscribe("final_answer", self._handle_final)
        self.bus.subscribe("clarification_request", self._handle_clarification)
        self._logger = logging.getLogger("orchestrator")

    def _register_agents(self) -> None:
        PlannerAgent("Planner", self.bus, self.llm).register()
        SearchAgent("Search", self.bus, self.llm, self.search_provider).register()
        SummarizerAgent("Summarizer", self.bus, self.llm).register()
        CriticAgent("Critic", self.bus, self.llm, max_rounds=self.max_rounds).register()
        WriterAgent("Writer", self.bus, self.llm).register()

    async def _handle_final(self, message: Message) -> None:
        self._logger.info("final answer received from=%s", message.sender)
        if self._final_future and not self._final_future.done():
            self._final_future.set_result(message.content)

    async def _handle_clarification(self, message: Message) -> None:
        self._logger.info("clarification request received from=%s", message.sender)
        if self._final_future and not self._final_future.done():
            self._final_future.set_result(message.content)

    async def run(self, query: str, timeout: float = 120.0) -> str:
        loop = asyncio.get_running_loop()
        self._final_future = loop.create_future()
        msg = self.bus.new_message("User", "user_query", query)
        await self.bus.publish(msg)
        return await asyncio.wait_for(self._final_future, timeout=timeout)
