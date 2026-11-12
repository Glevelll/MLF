"""Celery-приложение (prod): брокер/бэкенд Redis, периодические задачи через beat."""
from __future__ import annotations

import asyncio

from celery import Celery
from celery.schedules import crontab

from backend.core.config import get_settings

settings = get_settings()
celery_app = Celery("mlf", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json", result_serializer="json", accept_content=["json"], timezone="UTC",
    task_acks_late=True, worker_prefetch_multiplier=1,
    beat_schedule={
        "reindex-rag-nightly": {"task": "mlf.reindex_rag_task", "schedule": crontab(hour=3, minute=0)},
        "cleanup-cache-hourly": {"task": "mlf.cleanup_cache_task", "schedule": crontab(minute=15)},
        "cleanup-artifacts-daily": {"task": "mlf.cleanup_artifacts_task", "schedule": crontab(hour=4, minute=0)},
    },
)


def _run(coro):
    from backend.db.session import dispose_engine

    async def wrapper():
        try:
            return await coro
        finally:
            await dispose_engine()

    return asyncio.run(wrapper())


@celery_app.task(name="mlf.forecast_task")
def forecast_task(task_id: str) -> None:
    from backend.worker import jobs

    _run(jobs.forecast_task(task_id))


@celery_app.task(name="mlf.resume_task")
def resume_task(task_id: str, resolution: dict) -> None:
    from backend.worker import jobs

    _run(jobs.resume_task(task_id, resolution))


@celery_app.task(name="mlf.evaluate_model_task")
def evaluate_model_task(job_id: str, params: dict) -> dict:
    from backend.worker import jobs

    return _run(jobs.evaluate_model_task(job_id, params))


@celery_app.task(name="mlf.compare_models_task")
def compare_models_task(models: list[str], sample_size: int | None) -> dict:
    from backend.worker import jobs

    return _run(jobs.compare_models_task(models, sample_size))


@celery_app.task(name="mlf.reindex_rag_task")
def reindex_rag_task() -> dict:
    from backend.worker import jobs

    return _run(jobs.reindex_rag_task())


@celery_app.task(name="mlf.cleanup_cache_task")
def cleanup_cache_task() -> int:
    from backend.worker import jobs

    return _run(jobs.cleanup_cache_task())


@celery_app.task(name="mlf.cleanup_artifacts_task")
def cleanup_artifacts_task() -> int:
    from backend.worker import jobs

    return _run(jobs.cleanup_artifacts_task())
