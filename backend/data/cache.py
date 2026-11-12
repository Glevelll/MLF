"""Кэш (таблица cache в БД или Redis) со счётчиком попаданий."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import delete, select, update

from backend.utils.logging import get_logger

log = get_logger("cache")


def make_key(namespace: str, payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return f"{namespace}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:48]}"


class Cache(Protocol):
    async def get(self, key: str) -> dict[str, Any] | None: ...

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None: ...


class MemoryCache:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}
        self.hits: dict[str, int] = {}

    async def get(self, key: str) -> dict[str, Any] | None:
        if key in self.data:
            self.hits[key] = self.hits.get(key, 0) + 1
            return self.data[key]
        return None

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        self.data[key] = value


class DbCache:
    async def get(self, key: str) -> dict[str, Any] | None:
        from backend.db.models import CacheEntry
        from backend.db.session import session_scope

        async with session_scope() as s:
            row = (await s.execute(select(CacheEntry).where(CacheEntry.input_key == key))).scalar_one_or_none()
            if row is None:
                return None
            if row.expires_at is not None and row.expires_at.replace(tzinfo=UTC) < datetime.now(UTC):
                return None
            await s.execute(update(CacheEntry).where(CacheEntry.id == row.id).values(hits=CacheEntry.hits + 1))
            return row.output

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        from backend.db.models import CacheEntry
        from backend.db.session import session_scope

        expires = datetime.now(UTC) + timedelta(seconds=ttl_seconds) if ttl_seconds else None
        try:
            async with session_scope() as s:
                row = (await s.execute(select(CacheEntry).where(CacheEntry.input_key == key))).scalar_one_or_none()
                if row is None:
                    s.add(CacheEntry(input_key=key, namespace=key.split(":")[0], output=value, expires_at=expires))
                else:
                    row.output, row.expires_at = value, expires
        except Exception as e:  # pragma: no cover - гонка уникального ключа
            log.debug("cache set failed: %s", e)


class RedisCache:  # pragma: no cover - требует Redis
    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis

        self.r = aioredis.from_url(url)

    async def get(self, key: str) -> dict[str, Any] | None:
        raw = await self.r.get(key)
        if raw is None:
            return None
        await self.r.hincrby("cache:hits", key, 1)
        return json.loads(raw)

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        await self.r.set(key, json.dumps(value, default=str), ex=ttl_seconds)


async def cleanup_expired() -> int:
    from backend.db.models import CacheEntry
    from backend.db.session import session_scope

    async with session_scope() as s:
        res = await s.execute(delete(CacheEntry).where(CacheEntry.expires_at < datetime.now(UTC)))
        return res.rowcount or 0


def get_cache() -> Cache:
    from backend.core.config import get_settings

    s = get_settings()
    if s.cache_backend == "redis":  # pragma: no cover
        return RedisCache(s.redis_url)
    return DbCache()
