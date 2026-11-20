"""LLMManager: роутинг по провайдерам, retry/fallback, кэш, логирование, Tool Calling."""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from backend.core.llm.base import LLMError, LLMProvider, LLMResponse, Message, ToolCall
from backend.core.llm.parsing import extract_json, parse_tool_calls
from backend.core.llm.tokens import count_tokens
from backend.data.cache import Cache, make_key
from backend.utils.logging import get_logger
from backend.utils.tracing import Recorder, Tracer

if TYPE_CHECKING:
    from backend.core.config import Settings
    from backend.core.llm.tools import Tool

log = get_logger("llm")


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:32]


class ToolCallResult(BaseModel):
    calls: list[ToolCall] = Field(default_factory=list)
    mode: str = "default"  # native | text | default
    attempts: int = 0
    errors: list[str] = Field(default_factory=list)

    @property
    def call(self) -> ToolCall | None:
        return self.calls[0] if self.calls else None


class LLMManager:
    def __init__(self, providers: list[LLMProvider], recorder: Recorder | None = None, cache: Cache | None = None,
                 max_retries: int = 2, backoff: float = 0.5, request_id: str | None = None,
                 *, require_response: bool = False) -> None:
        if not providers:
            raise ValueError("нужен хотя бы один провайдер")
        self.providers = providers
        self.recorder = recorder
        self.cache = cache
        self.max_retries = max_retries
        self.backoff = backoff
        self.request_id = request_id
        self.require_response = require_response

    @property
    def model_name(self) -> str:
        return f"{self.providers[0].name}:{self.providers[0].model}"

    async def _record(self, **data: Any) -> None:
        if self.recorder is None:
            return
        span = Tracer.current()
        data.setdefault("request_id", self.request_id or (span.request_id if span else None))
        data.setdefault("span_id", span.id if span else None)
        await self.recorder.record_llm_call(data)

    async def chat(self, messages: list[Message], *, agent_name: str, tools: list[dict] | None = None,
                   json_mode: bool = False, temperature: float = 0.0, use_cache: bool = True) -> LLMResponse:
        payload = {"messages": [m.model_dump() for m in messages], "tools": tools, "json_mode": json_mode}
        prompt_hash = _hash(payload)
        key = make_key("llm", {"p": [p.name + p.model for p in self.providers], "h": prompt_hash})
        if self.cache is not None and use_cache and temperature == 0:
            hit = await self.cache.get(key)
            if hit is not None:
                resp = LLMResponse(**hit)
                resp.cached = True
                await self._record(agent_name=agent_name, model_name=resp.model, provider=resp.provider,
                                   prompt_hash=prompt_hash, response_hash=_hash(hit), cached=True, status="ok",
                                   parameters={"temperature": temperature})
                return resp
        prompt_text = "\n".join(m.content for m in messages) + (json.dumps(tools, ensure_ascii=False) if tools else "")
        last_err: Exception | None = None
        for provider in self.providers:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await provider.chat(messages, tools=tools, json_mode=json_mode, temperature=temperature)
                except LLMError as e:
                    last_err = e
                    await self._record(agent_name=agent_name, model_name=provider.model, provider=provider.name,
                                       prompt_hash=prompt_hash, status="error", error=str(e)[:500],
                                       parameters={"temperature": temperature, "attempt": attempt})
                    log.warning("LLM %s attempt %d failed: %s", provider.name, attempt, e)
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.backoff * 2**attempt)
                    continue
                estimated = resp.prompt_tokens is None or resp.completion_tokens is None
                if resp.prompt_tokens is None:
                    resp.prompt_tokens = count_tokens(prompt_text)
                if resp.completion_tokens is None:
                    resp.completion_tokens = count_tokens(
                        resp.content + "".join(json.dumps(c.arguments) + c.name for c in resp.tool_calls))
                total = resp.prompt_tokens + resp.completion_tokens
                span = Tracer.current()
                if span is not None:
                    span.add_tokens(total)
                dump = resp.model_dump()
                await self._record(agent_name=agent_name, model_name=resp.model or provider.model,
                                   provider=provider.name, prompt_hash=prompt_hash, response_hash=_hash(dump),
                                   prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                                   tokens_used=total, tokens_estimated=estimated, latency_ms=resp.latency_ms,
                                   status="ok", parameters={"temperature": temperature, "tools": bool(tools),
                                                            "json_mode": json_mode, "attempt": attempt})
                if self.cache is not None and temperature == 0:
                    await self.cache.set(key, dump, ttl_seconds=7 * 24 * 3600)
                return resp
        raise LLMError(f"все провайдеры недоступны: {last_err}")

    async def call_tool(self, messages: list[Message], tools: list[Tool], *, agent_name: str,
                        max_attempts: int = 3, multi: bool = False) -> ToolCallResult:
        """Строгий function calling -> парсинг текста -> повтор с уточнением (до max_attempts).

        Если валидный вызов не получен, возвращается результат с mode="default",
        и агент применяет правило по умолчанию.
        """
        by_name = {t.name: t for t in tools}
        schemas = [t.schema() for t in tools]
        msgs = list(messages)
        errors: list[str] = []
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await self.chat(msgs, agent_name=agent_name, tools=schemas, use_cache=attempt == 1)
            except LLMError as e:
                if self.require_response:
                    raise
                errors.append(f"attempt {attempt}: {e}")
                continue
            mode, calls = "native", resp.tool_calls
            if not calls:
                mode, calls = "text", parse_tool_calls(resp.content)
                if multi and not calls:
                    data = extract_json(resp.content)
                    if isinstance(data, dict) and data.get("tool_calls") == []:
                        return ToolCallResult(calls=[], mode=mode, attempts=attempt, errors=errors)
            valid: list[ToolCall] = []
            errs: list[str] = []
            for c in calls:
                tool = by_name.get(c.name)
                if tool is None:
                    errs.append(f"неизвестный инструмент '{c.name}'")
                    continue
                try:
                    args = tool.validate_args(c.arguments)
                except ValidationError as e:
                    errs.append(f"{c.name}: неверные аргументы: {e.errors()[:2]}")
                    continue
                valid.append(ToolCall(name=c.name, arguments=args))
            if not calls:
                errs.append("ответ не содержит вызова инструмента")
            if valid and (multi or not errs):
                return ToolCallResult(calls=valid if multi else valid[:1], mode=mode, attempts=attempt, errors=errors)
            errors.append(f"attempt {attempt}: " + "; ".join(errs))
            msgs = msgs + [
                Message(role="assistant", content=resp.content or json.dumps([c.model_dump() for c in calls])),
                Message(role="user", content=(
                    "Ответ не распознан: " + "; ".join(errs) + ". Верни СТРОГО JSON вида "
                    '{"name": "<инструмент>", "arguments": {...}} для одного из: ' + ", ".join(by_name))),
            ]
        return ToolCallResult(calls=[], mode="default", attempts=max_attempts, errors=errors)

    async def structured(self, messages: list[Message], schema: type[BaseModel], *, agent_name: str,
                         max_attempts: int = 3) -> tuple[BaseModel | None, int]:
        msgs = list(messages)
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await self.chat(msgs, agent_name=agent_name, json_mode=True, use_cache=attempt == 1)
            except LLMError:
                if self.require_response:
                    raise
                continue
            data = extract_json(resp.content)
            if isinstance(data, dict):
                try:
                    return schema.model_validate(data), attempt
                except ValidationError as e:
                    err = str(e.errors()[:2])
            else:
                err = "не найден JSON-объект"
            msgs = msgs + [Message(role="assistant", content=resp.content),
                           Message(role="user", content=f"Ошибка формата: {err}. Верни только JSON по схеме: "
                                                        f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}")]
        return None, max_attempts

    async def aclose(self) -> None:
        for p in self.providers:
            await p.aclose()


def make_provider(name: str, settings: Settings) -> LLMProvider:
    if name == "mock":
        from backend.core.llm.providers.mock import MockProvider

        return MockProvider()
    if name == "ollama":
        from backend.core.llm.providers.ollama import OllamaProvider

        return OllamaProvider(settings.llm_model, settings.ollama_url, settings.llm_timeout)
    if name == "hf":  # pragma: no cover - тяжёлая зависимость
        from backend.core.llm.providers.hf_local import HFLocalProvider

        return HFLocalProvider(settings.hf_model)
    if name == "openai":
        from backend.core.llm.providers.api import OpenAICompatibleProvider

        return OpenAICompatibleProvider(settings.openai_model, settings.openai_api_key, settings.openai_base_url,
                                        settings.llm_timeout)
    if name == "anthropic":
        from backend.core.llm.providers.api import AnthropicProvider

        return AnthropicProvider(settings.anthropic_model, settings.anthropic_api_key, settings.llm_timeout)
    raise ValueError(f"неизвестный провайдер {name}")


def build_llm_manager(settings: Settings, recorder: Recorder | None = None, cache: Cache | None = None,
                      request_id: str | None = None, *, require_response: bool = False) -> LLMManager:
    names = [settings.llm_provider] + [n for n in settings.llm_fallback_providers if n != settings.llm_provider]
    providers = [make_provider(n, settings) for n in names]
    return LLMManager(providers, recorder=recorder, cache=cache if settings.llm_cache_enabled else None,
                      max_retries=settings.llm_max_retries, request_id=request_id, require_response=require_response)
