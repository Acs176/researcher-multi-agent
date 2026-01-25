from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class Message:
    id: str
    sender: str
    type: str
    content: str
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    in_reply_to: Optional[str] = None
    created_at: float = field(default_factory=time.time)


Handler = Callable[[Message], Awaitable[None]]


class MessageBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Handler]] = {}
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger("bus")

    def subscribe(self, msg_type: str, handler: Handler) -> None:
        self._subscribers.setdefault(msg_type, []).append(handler)

    async def publish(self, message: Message) -> None:
        handlers = list(self._subscribers.get(message.type, []))
        self._logger.info(
            "publish type=%s sender=%s handlers=%s",
            message.type,
            message.sender,
            len(handlers),
        )
        for handler in handlers:
            await handler(message)

    @staticmethod
    def new_message(
        sender: str,
        msg_type: str,
        content: str,
        citations: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        in_reply_to: Optional[str] = None,
    ) -> Message:
        return Message(
            id=str(uuid.uuid4()),
            sender=sender,
            type=msg_type,
            content=content,
            citations=citations or [],
            metadata=metadata or {},
            in_reply_to=in_reply_to,
        )
