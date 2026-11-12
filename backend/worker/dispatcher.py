"""Диспетчер фоновых задач: inline (asyncio, локальная разработка) или Celery (prod)."""
from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from backend.core.config import get_settings
from backend.utils.logging import get_logger

log = get_logger("dispatcher")
_running: set[asyncio.Task] = set()


def _spawn(coro: Coroutine) -> None:
    t = asyncio.get_running_loop().create_task(coro)
    _running.add(t)
    t.add_done_callback(_running.discard)


def dispatch(name: str, *args: Any) -> str:
    """Отправляет задачу; возвращает режим исполнения."""
    settings = get_settings()
    if settings.task_backend == "celery":  # pragma: no cover - требует брокер
        from backend.worker.celery_app import celery_app

        celery_app.send_task(f"mlf.{name}", args=list(args))
        return "celery"
    from backend.worker import jobs

    _spawn(getattr(jobs, name)(*args))
    return "inline"


async def wait_all(timeout: float = 600) -> None:
    """Для тестов: дождаться завершения inline-задач."""
    while _running:
        await asyncio.wait(list(_running), timeout=timeout)
