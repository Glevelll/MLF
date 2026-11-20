"""Основные эндпоинты прогноза: создание задачи (202), статус, результат, reasoning, HITL."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path, PureWindowsPath
from typing import Literal

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user
from backend.api.schemas import HitlDecision, TaskAccepted, TaskStatusOut
from backend.core.config import get_settings
from backend.core.llm.base import LLMError
from backend.core.llm.providers.ollama import installed_qwen_models
from backend.core.llm.tools import default_registry
from backend.core.models import ForecastState, TaskStatus
from backend.data.storage import ArtifactStore
from backend.db.models import LLMCall, ReasoningLog, Task, User
from backend.db.session import get_session
from backend.tabular.datasets import read_table
from backend.tabular.preprocessing import STUDENT_GROUP
from backend.utils.tracing import build_tree
from backend.worker.dispatcher import dispatch

router = APIRouter(prefix="/api/v1/forecast", tags=["forecast"])
MAX_UPLOAD = 50 * 1024 * 1024
BUILTIN = ("uci_mat", "uci_por", "uci_both", "synthetic")


async def _load_task(task_id: str, user: User, session: AsyncSession) -> Task:
    task = await session.get(Task, task_id)
    if task is None or task.task_type == "evaluation":
        raise HTTPException(404, "задача не найдена")
    if task.owner_id != user.id and user.role not in ("editor", "admin"):
        raise HTTPException(403, "нет доступа к задаче")
    return task


async def _store_input(task_id: str, file: UploadFile | None, data_url: str | None, dataset: str | None) -> str:
    store = ArtifactStore()
    if file is not None:
        suffix = Path(file.filename or "data.csv").suffix.lower() or ".csv"
        if suffix not in (".csv", ".parquet", ".pq", ".txt"):
            raise HTTPException(400, "поддерживаются CSV и Parquet")
        content = await file.read()
        if len(content) > MAX_UPLOAD:
            raise HTTPException(413, "файл больше 50 МБ")
        path = store.path(task_id, f"input{suffix if suffix != '.txt' else '.csv'}")
        path.write_bytes(content)
        return str(path)
    if data_url:
        if not data_url.startswith(("http://", "https://")):
            raise HTTPException(400, "data_url должен быть http(s)")
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            r = await client.get(data_url)
        if r.status_code != 200 or len(r.content) > MAX_UPLOAD:
            raise HTTPException(400, f"не удалось скачать данные ({r.status_code})")
        suffix = ".parquet" if data_url.split("?")[0].endswith((".parquet", ".pq")) else ".csv"
        path = store.path(task_id, f"input{suffix}")
        path.write_bytes(r.content)
        return str(path)
    if dataset:
        from backend.tabular.datasets import generate_synthetic, load_uci

        if dataset not in BUILTIN:
            raise HTTPException(400, f"dataset должен быть одним из {BUILTIN}")
        df = generate_synthetic(1500, seed=7) if dataset == "synthetic" else load_uci(dataset.split("_", 1)[1])
        path = store.path(task_id, "input.parquet")
        df.to_parquet(path, index=False)
        return str(path)
    raise HTTPException(400, "передайте file, data_url или dataset")


@router.get("/agent-models", dependencies=[Depends(get_current_user)])
async def agent_models() -> dict:
    settings = get_settings()
    try:
        names = await installed_qwen_models(settings.ollama_url)
    except LLMError as exc:
        raise HTTPException(503, str(exc)) from exc
    preferred = settings.llm_model if settings.llm_model in names else next(iter(names), None)
    return {"models": names, "preferred": preferred}


@router.post("/", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_forecast(
    file: UploadFile | None = File(None),
    data_url: str | None = Form(None),
    dataset: str | None = Form(None),
    target_column: str | None = Form(None),
    task_type: Literal["tabular", "timeseries"] = Form("tabular"),
    problem_type: Literal["auto", "classification", "regression"] = Form("auto"),
    binarize_threshold: float | None = Form(None),
    positive_label: str | None = Form(None),
    id_column: str | None = Form(None),
    horizon: int = Form(0),
    freq: str | None = Form(None),
    max_iterations: int | None = Form(None),
    pass_mark: float | None = Form(None),
    agent_provider: Literal["mock", "ollama"] | None = Form(None),
    agent_model: str | None = Form(None),
    requested_model: str | None = Form(None),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> TaskAccepted:
    settings = get_settings()
    if requested_model:
        if task_type != "tabular" or problem_type == "auto":
            raise HTTPException(400, "Для выбора алгоритма укажите тип табличной задачи.")
        if requested_model not in default_registry().names(kind="model", task_type=task_type, problem_type=problem_type):
            raise HTTPException(400, "Выбранный алгоритм не подходит для этой задачи.")
    if agent_provider == "ollama":
        options = await agent_models()
        agent_model = agent_model or options["preferred"]
        if not agent_model or agent_model not in options["models"]:
            raise HTTPException(400, "Выберите установленную модель Qwen в списке Ollama.")
    elif agent_model:
        raise HTTPException(400, "Сначала выберите Ollama для запуска Qwen.")
    task = Task(owner_id=user.id, status=TaskStatus.pending, task_type=task_type)
    session.add(task)
    await session.flush()
    path = await _store_input(task.id, file, data_url, dataset)
    if task_type == "tabular" and not target_column:
        target_column = {"uci_mat": "G3", "uci_por": "G3", "uci_both": "G3", "synthetic": "dropout"}.get(dataset or "")
        if not target_column:
            raise HTTPException(400, "укажите target_column")
    try:
        cols = list(pd.read_parquet(path).columns) if path.endswith(".parquet") else None
        if cols is not None and target_column and target_column not in cols:
            raise HTTPException(400, f"колонка {target_column} не найдена")
    except HTTPException:
        raise
    except Exception:  # pragma: no cover - проверка формата будет у DataAnalyst
        pass
    source = PureWindowsPath(file.filename).name if file is not None and file.filename else dataset or Path(path).name
    state = ForecastState(
        task_id=task.id, series_id=dataset or Path(path).name, horizon=horizon, freq=freq, task_type=task_type,
        target_column=target_column, problem_type=problem_type, binarize_threshold=binarize_threshold,
        positive_label=positive_label, id_column=id_column, data_path=path,
        max_iterations=max_iterations or settings.max_iterations,
        agent_provider=agent_provider, agent_model=agent_model, requested_model=requested_model,
        metadata={"input_source": source, **({"pass_mark": pass_mark} if pass_mark is not None else {})},
    )
    task.params = {k: v for k, v in state.model_dump(mode="json").items()
                   if k in ("target_column", "problem_type", "binarize_threshold", "horizon", "task_type", "series_id")}
    task.state = state.model_dump(mode="json")
    await session.commit()
    mode = dispatch("forecast_task", task.id)
    return TaskAccepted(task_id=task.id, status=task.status, execution=mode)


@router.get("/", response_model=list[TaskStatusOut])
async def list_tasks(status_filter: str | None = None, limit: int = 50, user: User = Depends(get_current_user),
                     session: AsyncSession = Depends(get_session)) -> list[TaskStatusOut]:
    q = select(Task).where(Task.task_type != "evaluation").order_by(Task.created_at.desc()).limit(limit)
    if user.role not in ("editor", "admin"):
        q = q.where(Task.owner_id == user.id)
    if status_filter:
        q = q.where(Task.status == status_filter)
    rows = (await session.execute(q)).scalars().all()
    return [_status(t) for t in rows]


def _status(t: Task) -> TaskStatusOut:
    st = t.state or {}
    return TaskStatusOut(task_id=t.id, status=t.status, task_type=t.task_type, created_at=t.created_at,
                         updated_at=t.updated_at, source=(st.get("metadata") or {}).get("input_source")
                         or st.get("series_id"), target_column=st.get("target_column"),
                         agent_provider=st.get("agent_provider"), agent_model=st.get("agent_model"),
                         iteration=st.get("iteration"), hitl_reason=st.get("hitl_reason"),
                         error=(t.error or "")[:500] or None)


@router.get("/{task_id}/status", response_model=TaskStatusOut)
async def get_status(task_id: str, user: User = Depends(get_current_user),
                     session: AsyncSession = Depends(get_session)) -> TaskStatusOut:
    return _status(await _load_task(task_id, user, session))


def source_preview(path: Path, target: str, subset: str, offset: int, limit: int) -> dict:
    frame = read_table(path).drop(columns=[STUDENT_GROUP], errors="ignore")
    if target not in frame:
        raise ValueError("Целевая колонка отсутствует в исходных данных")
    known = frame[target].notna()
    known_count = int(known.sum())
    if subset == "known":
        selected = frame[known]
    elif subset == "new":
        selected = frame[~known]
    else:
        selected = frame
    page = selected.iloc[offset:offset + limit]
    return {"known_rows": known_count, "new_rows": len(frame) - known_count, "matching_rows": len(selected),
            "columns": list(frame.columns), "rows": json.loads(page.to_json(orient="values", date_format="iso"))}


@router.get("/{task_id}/data")
async def get_input_data(task_id: str, subset: Literal["all", "known", "new"] = "all",
                         offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=50),
                         user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)) -> dict:
    task = await _load_task(task_id, user, session)
    state = ForecastState.model_validate(task.state)
    if state.task_type != "tabular" or not state.target_column:
        raise HTTPException(409, "Просмотр доступен для табличных прогнозов")
    if not state.data_path:
        raise HTTPException(404, "Исходная таблица не найдена")
    try:
        preview = await asyncio.to_thread(source_preview, Path(state.data_path), state.target_column, subset, offset, limit)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Исходная таблица не найдена") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(422, "Не удалось прочитать исходную таблицу") from exc
    features = state.metadata.get("working_features") or state.feature_columns
    return {**preview, "source": state.metadata.get("input_source") or state.series_id,
            "target_column": state.target_column, "problem_type": state.problem_type,
            "model": state.model_type, "features": features}


@router.get("/{task_id}/result")
async def get_result(task_id: str, preview: int = 20, user: User = Depends(get_current_user),
                     session: AsyncSession = Depends(get_session)) -> dict:
    task = await _load_task(task_id, user, session)
    if task.status not in (TaskStatus.completed, TaskStatus.waiting_for_human):
        raise HTTPException(409, f"результат недоступен, статус: {task.status}")
    st = ForecastState.model_validate(task.state)
    out = {
        "task_id": task_id, "status": task.status, "problem_type": st.problem_type, "model_type": st.model_type,
        "model_params": st.model_params, "selection_mode": st.selection_mode, "metrics": st.metrics,
        "decision_threshold": st.decision_threshold, "critic_score": st.critic_score,
        "critic_reasoning": st.critic_reasoning, "critic_details": st.critic_details,
        "issues": [i.model_dump() for i in st.validation_issues], "iterations": st.iteration,
        "feature_importance": st.metadata.get("feature_importance"), "feature_ops": st.feature_ops,
        "dropped_features": st.dropped_features, "editor_actions": [a.model_dump() for a in st.editor_actions],
        "rag_cases": st.rag_context, "risk_counts": st.metadata.get("risk_counts"),
        "eda": (st.diagnostics or {}).get("eda"), "predictions_path": st.predictions_path,
        "tokens_used": st.tokens_used, "hitl_reason": st.hitl_reason, "hitl_resolution": st.hitl_resolution,
        "available_models": [], "predictions_preview": [],
    }
    out["available_models"] = default_registry().names(kind="model", task_type=st.task_type,
                                                       problem_type=st.problem_type)
    if st.predictions_path and Path(st.predictions_path).exists() and st.predictions_path.endswith(".csv"):
        df = pd.read_csv(st.predictions_path)
        col = "risk_probability" if st.problem_type == "classification" else "predicted_score"
        asc = st.problem_type != "classification"
        if col in df.columns:
            df = df.sort_values(col, ascending=asc)
        out["predictions_preview"] = df.head(preview).fillna("").to_dict("records")
    return out


@router.get("/{task_id}/predictions")
async def download_predictions(task_id: str, user: User = Depends(get_current_user),
                               session: AsyncSession = Depends(get_session)):
    task = await _load_task(task_id, user, session)
    path = (task.state or {}).get("predictions_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "прогноз ещё не сформирован")
    return FileResponse(path, filename=f"predictions_{task_id}{Path(path).suffix}")


@router.get("/{task_id}/reasoning")
async def get_reasoning(task_id: str, user: User = Depends(get_current_user),
                        session: AsyncSession = Depends(get_session)) -> dict:
    await _load_task(task_id, user, session)
    rows = (await session.execute(select(ReasoningLog).where(ReasoningLog.request_id == task_id))).scalars().all()
    calls = (await session.execute(select(LLMCall).where(LLMCall.request_id == task_id)
                                   .order_by(LLMCall.created_at))).scalars().all()
    spans = [{c.name: getattr(r, c.name) for c in ReasoningLog.__table__.columns} for r in rows]
    for s in spans:
        for k in ("started_at", "finished_at"):
            s[k] = s[k].isoformat() if s[k] else None
    return {
        "task_id": task_id, "tree": build_tree(spans), "n_spans": len(spans),
        "llm_calls": [{"agent": c.agent_name, "provider": c.provider, "model": c.model_name, "tokens": c.tokens_used,
                       "estimated": c.tokens_estimated, "latency_ms": round(c.latency_ms, 1), "status": c.status,
                       "cached": c.cached, "span_id": c.span_id, "created_at": c.created_at.isoformat()}
                      for c in calls],
        "total_tokens": sum(c.tokens_used for c in calls),
    }


@router.post("/{task_id}/hitl", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED)
async def submit_hitl(task_id: str, decision: HitlDecision, user: User = Depends(get_current_user),
                      session: AsyncSession = Depends(get_session)) -> TaskAccepted:
    task = await _load_task(task_id, user, session)
    if task.status != TaskStatus.waiting_for_human:
        raise HTTPException(409, f"задача не ожидает решения человека (статус {task.status})")
    st = ForecastState.model_validate(task.state)
    if decision.action == "choose_model":
        allowed = default_registry().names(kind="model", task_type=st.task_type, problem_type=st.problem_type)
        if decision.model_type not in allowed:
            raise HTTPException(400, f"model_type должен быть одним из {allowed}")
    if decision.action == "change_threshold" and decision.threshold is None:
        raise HTTPException(400, "укажите threshold")
    if decision.action == "change_horizon" and decision.horizon is None:
        raise HTTPException(400, "укажите horizon")
    resolution = {**decision.model_dump(exclude_none=True), "user_id": user.id}
    from backend.db.models import HitlReview

    reviews = (await session.execute(select(HitlReview).where(HitlReview.task_id == task_id,
                                                              HitlReview.kind == "task_quality",
                                                              HitlReview.status == "pending"))).scalars().all()
    from backend.hitl.active_learning import resolve

    for r in reviews:
        await resolve(session, r, "rejected" if decision.action == "reject" else "confirmed", resolution, user.id)
    task.status = TaskStatus.running if decision.action != "reject" else task.status
    await session.commit()
    mode = dispatch("resume_task", task_id, resolution)
    return TaskAccepted(task_id=task_id, status="running" if decision.action != "reject" else "failed", execution=mode)
