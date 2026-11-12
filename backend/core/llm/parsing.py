"""Разбор текстового ответа LLM в вызов инструмента / JSON (fallback для слабых моделей)."""
from __future__ import annotations

import json
import re
from typing import Any

from backend.core.llm.base import ToolCall

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any | None:
    """Находит первый корректный JSON-объект/массив в тексте."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            while start != -1:
                depth, in_str, esc = 0, False, False
                for i in range(start, len(cand)):
                    ch = cand[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(cand[start : i + 1])
                            except (json.JSONDecodeError, ValueError):
                                break
                start = cand.find(opener, start + 1)
    return None


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Поддерживаемые формы: {"name":..,"arguments":{..}}, {"tool":..,"args":..},
    {"function":{"name":..,"arguments":..}}, список таких объектов, {"tool_calls":[..]}."""
    data = extract_json(text)
    if data is None:
        return []
    if isinstance(data, dict) and "tool_calls" in data:
        data = data["tool_calls"]
    items = data if isinstance(data, list) else [data]
    calls: list[ToolCall] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if "function" in it and isinstance(it["function"], dict):
            it = it["function"]
        name = it.get("name") or it.get("tool") or it.get("tool_name")
        args = it.get("arguments", it.get("args", it.get("parameters", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(name, str) and isinstance(args, dict):
            calls.append(ToolCall(name=name, arguments=args))
    return calls
