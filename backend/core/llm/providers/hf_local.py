"""Локальная causal LM через transformers.pipeline (запасной вариант для слабых машин).

Tool calling эмулируется: описания инструментов кладутся в системный промпт,
ответ разбирается парсером JSON (см. LLMManager.call_tool, fallback-режим).
"""
from __future__ import annotations

import asyncio
import json
import time

from backend.core.llm.base import LLMError, LLMProvider, LLMResponse
from backend.core.llm.parsing import parse_tool_calls


class HFLocalProvider(LLMProvider):
    name = "hf"

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens: int = 512) -> None:
        super().__init__(model)
        self.max_new_tokens = max_new_tokens
        self._pipe = None
        self._lock = asyncio.Lock()

    def _load(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise LLMError("transformers не установлен: pip install .[hf]") from e
            self._pipe = pipeline("text-generation", model=self.model, device_map="auto")
        return self._pipe

    async def chat(self, messages, *, tools=None, json_mode=False, temperature=0.0) -> LLMResponse:
        msgs = [{"role": m.role if m.role != "tool" else "user", "content": m.content} for m in messages]
        if tools:
            spec = json.dumps([t["function"] for t in tools], ensure_ascii=False)
            msgs.insert(0, {"role": "system", "content":
                            "Доступные инструменты (JSON Schema):\n" + spec +
                            '\nОтветь ТОЛЬКО JSON вида {"name": "<tool>", "arguments": {...}}.'})
        t0 = time.perf_counter()
        async with self._lock:
            pipe = await asyncio.to_thread(self._load)
            out = await asyncio.to_thread(
                pipe, msgs, max_new_tokens=self.max_new_tokens, do_sample=temperature > 0,
                temperature=max(temperature, 1e-3), return_full_text=False,
            )
        text = out[0]["generated_text"]
        if isinstance(text, list):
            text = text[-1].get("content", "")
        calls = parse_tool_calls(text) if tools else []
        tok = pipe.tokenizer
        p_tok = len(tok.apply_chat_template(msgs, tokenize=True))
        c_tok = len(tok.encode(text))
        return LLMResponse(content=text, tool_calls=calls, model=self.model, provider=self.name,
                           prompt_tokens=p_tok, completion_tokens=c_tok,
                           latency_ms=(time.perf_counter() - t0) * 1000)
