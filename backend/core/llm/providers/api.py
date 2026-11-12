"""Внешние API: OpenAI-совместимые (OpenAI, Groq, OpenRouter, vLLM) и Anthropic."""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from backend.core.llm.base import LLMError, LLMProvider, LLMResponse, Message, ToolCall


class OpenAICompatibleProvider(LLMProvider):
    """Подходит для OpenAI, Groq, OpenRouter и vLLM (OpenAI-совместимый сервер)."""

    name = "openai"

    def __init__(self, model: str, api_key: str | None, base_url: str = "https://api.openai.com/v1",
                 timeout: float = 60.0, client: httpx.AsyncClient | None = None, name: str = "openai") -> None:
        super().__init__(model)
        self.name = name
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.AsyncClient(timeout=timeout, headers=headers)

    @staticmethod
    def _msg(m: Message) -> dict[str, Any]:
        d: dict[str, Any] = {"role": m.role if m.role != "tool" else "user", "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = [
                {"id": f"call_{i}", "type": "function",
                 "function": {"name": t.name, "arguments": json.dumps(t.arguments)}}
                for i, t in enumerate(m.tool_calls)
            ]
        return d

    async def chat(self, messages, *, tools=None, json_mode=False, temperature=0.0) -> LLMResponse:
        body: dict[str, Any] = {"model": self.model, "messages": [self._msg(m) for m in messages],
                                "temperature": temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if json_mode and not tools:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            r = await self.client.post(f"{self.base_url}/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMError(f"{self.name}: {e}") from e
        calls = []
        for tc in choice.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args))
        usage = data.get("usage") or {}
        return LLMResponse(content=choice.get("content") or "", tool_calls=calls, model=data.get("model", self.model),
                           provider=self.name, prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"),
                           latency_ms=(time.perf_counter() - t0) * 1000)

    async def aclose(self) -> None:
        await self.client.aclose()


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None, timeout: float = 60.0,
                 client: httpx.AsyncClient | None = None) -> None:
        super().__init__(model)
        headers = {"x-api-key": api_key or "", "anthropic-version": "2023-06-01"}
        self.client = client or httpx.AsyncClient(timeout=timeout, headers=headers)

    async def chat(self, messages, *, tools=None, json_mode=False, temperature=0.0) -> LLMResponse:
        system = "\n".join(m.content for m in messages if m.role == "system")
        msgs = [{"role": "assistant" if m.role == "assistant" else "user", "content": m.content or "."}
                for m in messages if m.role != "system"]
        body: dict[str, Any] = {"model": self.model, "max_tokens": 1024, "messages": msgs, "temperature": temperature}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                              "input_schema": t["function"]["parameters"]} for t in tools]
        t0 = time.perf_counter()
        try:
            r = await self.client.post("https://api.anthropic.com/v1/messages", json=body)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            raise LLMError(f"anthropic: {e}") from e
        text, calls = [], []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(name=block.get("name", ""), arguments=block.get("input") or {}))
        usage = data.get("usage") or {}
        return LLMResponse(content="".join(text), tool_calls=calls, model=data.get("model", self.model),
                           provider=self.name, prompt_tokens=usage.get("input_tokens"),
                           completion_tokens=usage.get("output_tokens"),
                           latency_ms=(time.perf_counter() - t0) * 1000)

    async def aclose(self) -> None:
        await self.client.aclose()
