"""Фильтры по метаданным: {"field": value} | {"field": {"gte": .., "lte": .., "in": [..], "ne": ..}}."""
from __future__ import annotations

from typing import Any


def match(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, cond in filters.items():
        v = payload.get(key)
        if isinstance(cond, dict):
            if "eq" in cond and v != cond["eq"]:
                return False
            if "ne" in cond and v == cond["ne"]:
                return False
            if "in" in cond and v not in cond["in"]:
                return False
            if "gte" in cond and (v is None or v < cond["gte"]):
                return False
            if "lte" in cond and (v is None or v > cond["lte"]):
                return False
        elif v != cond:
            return False
    return True
