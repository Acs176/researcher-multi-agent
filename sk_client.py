from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional
from semantic_kernel.kernel import KernelArguments

try:
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

    try:
        from semantic_kernel.functions.kernel_function_from_prompt import KernelFunctionFromPrompt
    except Exception:  # pragma: no cover - optional
        KernelFunctionFromPrompt = None
except Exception:  # pragma: no cover - fallback for missing dependency
    Kernel = None
    OpenAIChatCompletion = None
    OpenAIChatPromptExecutionSettings = None
    KernelFunctionFromPrompt = None


@dataclass
class SKConfig:
    model: str
    api_key: str
    org_id: Optional[str] = None
    service_id: str = "chat"
    # temperature: float = 0.2
    max_tokens: int = 900
    max_completion_tokens: Optional[int] = None


class SKClient:
    def __init__(self, config: SKConfig) -> None:
        if Kernel is None:
            raise RuntimeError("semantic-kernel is not installed.")
        self._config = config
        self._logger = logging.getLogger("sk_client")
        self.kernel = Kernel()
        service = OpenAIChatCompletion(
            service_id=config.service_id,
            ai_model_id=config.model,
            api_key=config.api_key,
            org_id=config.org_id,
        )
        self.kernel.add_service(service)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[dict] = None,
    ) -> str:
        if OpenAIChatPromptExecutionSettings is None:
            raise RuntimeError("semantic-kernel OpenAI settings unavailable.")
        settings = OpenAIChatPromptExecutionSettings(
            service_id=self._config.service_id,
            # temperature=self._config.temperature, ONLY DEFAULT (1) SUPPORTED WITH GPT5
            max_completion_tokens=self._config.max_completion_tokens or self._config.max_tokens,
        )
        if response_format:
            settings.response_format = response_format
        prompt = system_prompt + "\n\nUser:\n{{$input}}"
        self._logger.debug("llm request model=%s response_format=%s", self._config.model, response_format)
        self._logger.debug("llm system_prompt=%s", _truncate(system_prompt, 2000))
        self._logger.debug("llm user_prompt=%s", _truncate(user_prompt, 2000))
        self._logger.debug("llm prompt_template=%s", _truncate(prompt, 2000))
        function = self._build_function(prompt, settings)
        result = await self._invoke(function, user_prompt)
        text = self._extract_text(result)
        if not text:
            self._logger.error(
                "Empty LLM output. result_type=%s repr=%s",
                type(result).__name__,
                repr(result),
            )
        return text

    def _build_function(self, prompt: str, settings):
        if hasattr(self.kernel, "create_function_from_prompt"):
            return self.kernel.create_function_from_prompt(
                prompt=prompt,
                execution_settings=settings,
            )
        if KernelFunctionFromPrompt is None:
            raise RuntimeError(
                "Semantic Kernel does not support create_function_from_prompt "
                "and KernelFunctionFromPrompt is unavailable."
            )
        return KernelFunctionFromPrompt(
            prompt=prompt,
            prompt_execution_settings=settings,
            function_name="chat_prompt",
        )

    async def _invoke(self, function, user_prompt: str):

        args = KernelArguments()
        args["input"] = user_prompt
        try:
            return await self.kernel.invoke(function, arguments=args)
        except TypeError:
            print("ERROR PASSING ARGS")
            return await self.kernel.invoke(function, user_prompt)

    def _extract_text(self, result) -> str:
        if result is None:
            return ""
        for attr in ("value", "content", "text"):
            if hasattr(result, attr):
                value = getattr(result, attr)
                return self._coerce_text(value)
        return self._coerce_text(result)

    def _coerce_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            first = value[0]
            for attr in ("content", "text", "value"):
                if hasattr(first, attr):
                    return self._coerce_text(getattr(first, attr))
            return str(first)
        if isinstance(value, dict) and "content" in value:
            return str(value["content"])
        return str(value)


def config_from_env() -> SKConfig:
    model = os.getenv("SK_MODEL", "gpt-5-mini")
    api_key = os.getenv("OPENAI_API_KEY", "")
    org_id = os.getenv("OPENAI_ORG_ID")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    max_tokens = int(os.getenv("SK_MAX_TOKENS", "900"))
    max_completion_tokens = os.getenv("SK_MAX_COMPLETION_TOKENS")
    return SKConfig(
        model=model,
        api_key=api_key,
        org_id=org_id,
        max_tokens=max_tokens,
        max_completion_tokens=int(max_completion_tokens) if max_completion_tokens else None,
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
