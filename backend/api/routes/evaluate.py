"""Оценка качества: сравнение моделей, пакетная оценка режимов, результаты."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user
from backend.api.schemas import BatchEvalRequest, TaskAccepted
from backend.core.llm.tools import default_registry
from backend.db.models import EvaluationResult, ModelComparison, Task, User
from backend.db.session import get_session
from backend.worker import jobs
from backend.worker.dispatcher import dispatch

router = APIRouter(prefix="/api/v1/evaluate", tags=["evaluate"])


@router.get("/compare")
async def compare(models: str = Query("fit_logistic_regression,fit_random_forest,fit_tabular_xgboost"),
                  sample_size: int | None = Query(None, ge=50), dataset: str = "uci_both",
                  target: str = "G3", binarize_threshold: float | None = 10.0,
                  _: User = Depends(get_current_user)) -> dict:
    names = [m.strip() for m in models.split(",") if m.strip()]
    reg = default_registry()
    unknown = [m for m in names if m not in reg]
    if unknown or len(names) < 2:
        raise HTTPException(400, f"нужно ≥2 модели из реестра; неизвестные: {unknown}")
    if dataset not in ("uci_mat", "uci_por", "uci_both"):
        raise HTTPException(400, "dataset: uci_mat | uci_por | uci_both")
    ptype = "classification" if binarize_threshold is not None else "regression"
    allowed = reg.names(kind="model", task_type="tabular", problem_type=ptype)
    bad = [m for m in names if m not in allowed]
    if bad:
        raise HTTPException(400, f"модели {bad} не подходят для задачи {ptype}")
    return await jobs.compare_models_task(names, sample_size, dataset, target, binarize_threshold)


@router.post("/batch", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
async def batch(req: BatchEvalRequest, user: User = Depends(get_current_user),
                session: AsyncSession = Depends(get_session)) -> TaskAccepted:
    job = Task(owner_id=user.id, status="pending", task_type="evaluation", params=req.model_dump())
    session.add(job)
    await session.commit()
    mode = dispatch("evaluate_model_task", job.id, req.model_dump())
    return TaskAccepted(task_id=job.id, status="pending", execution=mode)


@router.get("/batch/{job_id}")
async def batch_status(job_id: str, _: User = Depends(get_current_user),
                       session: AsyncSession = Depends(get_session)) -> dict:
    job = await session.get(Task, job_id)
    if job is None or job.task_type != "evaluation":
        raise HTTPException(404, "задание не найдено")
    return {"job_id": job_id, "status": job.status, "params": job.params, "error": job.error,
            "summary": (job.state or {}).get("summary"), "rows": (job.state or {}).get("rows", [])[:500]}


@router.get("/results")
async def results(limit: int = 50, _: User = Depends(get_current_user),
                  session: AsyncSession = Depends(get_session)) -> dict:
    ev = (await session.execute(select(EvaluationResult).order_by(EvaluationResult.created_at.desc())
                                .limit(limit))).scalars().all()
    cmp_ = (await session.execute(select(ModelComparison).order_by(ModelComparison.created_at.desc())
                                  .limit(limit))).scalars().all()
    return {
        "evaluations": [{"run_id": e.run_id, "mode": e.mode, "sample_size": e.sample_size, "metrics": e.metrics_json,
                         "time": e.evaluation_time, "created_at": e.created_at.isoformat()} for e in ev],
        "comparisons": [{"id": c.id, "models": c.models_compared, "sample_size": c.sample_size, "winner": c.winner,
                         "metric": c.results_json.get("metric"),
                         "scores": {m: r.get("fold_mean") for m, r in c.results_json.get("results", {}).items()},
                         "created_at": c.created_at.isoformat()} for c in cmp_],
    }
