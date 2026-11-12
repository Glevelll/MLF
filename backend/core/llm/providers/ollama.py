"""Асинхронный клиент Ollama (/api/chat) с поддержкой tool calling."""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from backend.core.llm.base import LLMError, LLMProvider, LLMResponse, Message, ToolCall


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: float = 120.0,
                 client: httpx.AsyncClient | None = None) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=timeout)

    @staticmethod
    def _msg(m: Message) -> dict[str, Any]:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = [{"function": {"name": t.name, "arguments": t.arguments}} for t in m.tool_calls]
        return d

    async def chat(self, messages, *, tools=None, json_mode=False, temperature=0.0) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [self._msg(m) for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            body["tools"] = tools
        if json_mode:
            body["format"] = "json"
        t0 = time.perf_counter()
        try:
            r = await self.client.post(f"{self.base_url}/api/chat", json=body)
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            raise LLMError(f"ollama: {e}") from e
        msg = data.get("message") or {}
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args or {}))
        return LLMResponse(
            content=msg.get("content", "") or "",
            tool_calls=calls,
            model=data.get("model", self.model),
            provider=self.name,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    async def aclose(self) -> None:
        await self.client.aclose()
