"""Асинхронные задачи (исполняются inline или Celery-воркером)."""
from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

from backend.core.config import get_settings
from backend.db.models import EvaluationResult, ModelComparison, Task
from backend.db.session import session_scope
from backend.utils.logging import get_logger

log = get_logger("jobs")


async def forecast_task(task_id: str) -> None:
    from backend.worker.pipeline import run_forecast

    await run_forecast(task_id)


async def resume_task(task_id: str, resolution: dict[str, Any]) -> None:
    from backend.worker.pipeline import resume_forecast

    await resume_forecast(task_id, resolution)


async def evaluate_model_task(job_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Пакетная оценка режимов на наборе спецификаций (как scripts/run_eval.py, но в фоне)."""
    from backend.evaluation.runner import evaluate_spec, summarize
    from backend.evaluation.specs import make_specs
    from backend.retrieval.index_manager import get_retriever

    async with session_scope() as s:
        job = await s.get(Task, job_id)
        job.status = "running"
    t0 = time.perf_counter()
    try:
        specs = make_specs(int(params.get("num_datasets", 5)), "eval", int(params.get("seed", 0)))
        rows: list[dict[str, Any]] = []
        cache: dict = {}
        for spec in specs:
            rows += await evaluate_spec(spec, modes=params.get("modes", ["baseline", "+rag"]),
                                        fractions=params.get("fractions", [1.0]),
                                        holdouts=params.get("holdouts", [0.2]), retriever=get_retriever(),
                                        uci_cache=cache)
        summary = summarize(rows)
        elapsed = time.perf_counter() - t0
        async with session_scope() as s:
            for g in summary["groups"]:
                s.add(EvaluationResult(run_id=job_id, model_name=g["mode"], mode=g["mode"],
                                       sample_size=int(params.get("num_datasets", 5)),
                                       metrics_json=g, evaluation_time=elapsed))
            job = await s.get(Task, job_id)
            job.status = "completed"
            job.completed_at = datetime.now(UTC)
            job.state = {"summary": summary, "rows": [{k: v for k, v in r.items() if k != "_loss"} for r in rows]}
        return summary
    except Exception as e:
        log.exception("evaluation failed")
        async with session_scope() as s:
            job = await s.get(Task, job_id)
            job.status, job.error = "failed", str(e)
        return {}


async def compare_models_task(models: list[str], sample_size: int | None, dataset: str = "uci_both",
                              target: str = "G3", binarize_threshold: float | None = 10.0) -> dict[str, Any]:
    from backend.evaluation.runner import compare_models
    from backend.tabular.datasets import load_uci
    from backend.tabular.preprocessing import prepare_dataframe

    subj = dataset.split("_", 1)[1] if dataset.startswith("uci") else "both"
    df = load_uci(subj)
    data = prepare_dataframe(df, target, "auto", binarize_threshold)
    res = await asyncio.to_thread(compare_models, data, models, sample_size)
    async with session_scope() as s:
        row = ModelComparison(models_compared=models, sample_size=res["sample_size"], results_json=res,
                              winner=res["winner"])
        s.add(row)
    return res


async def reindex_rag_task() -> dict[str, int]:
    from backend.retrieval.index_manager import rebuild_from_db

    return await rebuild_from_db()


async def cleanup_cache_task() -> int:
    from backend.data.cache import cleanup_expired

    return await cleanup_expired()


async def cleanup_artifacts_task(max_age_days: int = 30) -> int:
    import shutil

    root = get_settings().artifacts_dir
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for d in root.glob("eval_*"):
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed
