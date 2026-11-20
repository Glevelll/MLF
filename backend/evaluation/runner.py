"""Оценка режимов baseline / +rag / +agents / +foundation и сравнение моделей."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from backend.core.config import Settings, get_settings
from backend.core.llm.tools import DEFAULT_MODEL, ToolRegistry, default_registry
from backend.core.metrics import clean, compute_metrics, primary_metric
from backend.data.embeddings import get_extractor
from backend.data.knowledge_base import describe_dataset_case
from backend.evaluation.specs import DatasetSpec, build_dataset, prepare_kwargs
from backend.evaluation.stats import compare_paired, diebold_mariano
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.index_manager import build_retriever
from backend.tabular.preprocessing import STUDENT_GROUP, PreparedData, prepare_dataframe
from backend.tabular.training import cv_splits, holdout_indices, make_pipeline
from backend.utils.logging import get_logger

log = get_logger("eval")
MODES = ["baseline", "+rag", "+agents", "+foundation"]


def build_evaluation_retriever(specs: list[DatasetSpec], settings: Settings) -> HybridRetriever:
    """Independent model-selection cases for benchmark runs on UCI and synthetic data."""
    path = settings.kb_dir / "dataset_cases.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} не найден — запустите scripts/generate_knowledge_base.py")
    eval_seeds = {spec.seed for spec in specs if spec.source == "synthetic"}
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        spec = case.get("meta", {}).get("spec", {})
        if (case.get("kind") == "dataset_case" and spec.get("source") == "synthetic"
                and isinstance(spec.get("seed"), int) and spec["seed"] not in eval_seeds):
            cases.append(case)
    if not cases:
        raise ValueError("Для оценки RAG нужны независимые синтетические кейсы выбора модели")
    local = settings.model_copy(update={"vector_backend": "memory"})
    return build_retriever(cases, settings=local, persist=False)


def subset(data: PreparedData, idx: np.ndarray) -> PreparedData:
    return PreparedData(X=data.X.iloc[idx].reset_index(drop=True), y=data.y.iloc[idx].reset_index(drop=True),
                        problem_type=data.problem_type, numeric_cols=data.numeric_cols,
                        categorical_cols=data.categorical_cols, ids=data.ids.iloc[idx].reset_index(drop=True),
                        groups=data.groups.iloc[idx].reset_index(drop=True) if data.groups is not None else None)


def split_holdout(data: PreparedData, holdout: float, seed: int) -> tuple[PreparedData, PreparedData]:
    tr, te = holdout_indices(data, holdout, seed)
    return subset(data, tr), subset(data, te)


def subsample(data: PreparedData, frac: float, seed: int) -> PreparedData:
    """Стратифицированная подвыборка (сохраняет пропорции классов)."""
    if frac >= 1.0:
        return data
    keep, _ = holdout_indices(data, frac, seed, test=False)
    return subset(data, np.sort(keep))


def per_sample_loss(problem_type: str, y, pred) -> np.ndarray:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return (y - pred) ** 2


def fit_predict(registry: ToolRegistry, tool_name: str, args: dict, train: PreparedData, test: PreparedData,
                seed: int) -> np.ndarray:
    tool = registry.get(tool_name)
    args = tool.validate_args(args)
    pipe = make_pipeline(tool.builder(args, train.problem_type, seed), train, scale=tool.scale).fit(train.X, train.y)
    if train.problem_type == "classification":
        return pipe.predict_proba(test.X)[:, 1]
    return pipe.predict(test.X)


def rag_choice(retriever: HybridRetriever | None, train: PreparedData, k: int = 5,
               available: list[str] | None = None) -> tuple[str, dict, list[str]]:
    """Режим +rag без LLM: голосование лучших моделей похожих кейсов (вес 1/rank)."""
    ptype = train.problem_type
    if retriever is None or retriever.size == 0:
        return DEFAULT_MODEL[ptype], {}, []
    meta = get_extractor("tabular").extract(X=train.X, y=train.y, problem_type=ptype,
                                            numeric_cols=train.numeric_cols, categorical_cols=train.categorical_cols)
    docs = retriever.search(describe_dataset_case(meta, "", {}, "запрос"), kind="dataset_case",
                            numeric=get_extractor("tabular").vector(meta), k=k, filters={"problem_type": ptype})
    votes: Counter[str] = Counter()
    params: dict[str, dict] = {}
    for rank, d in enumerate(docs):
        m = d.payload.get("best_model")
        if m and (available is None or m in available):
            votes[m] += 1 / (rank + 1)
            params.setdefault(m, d.payload.get("hyperparams") or {})
    if not votes:
        return DEFAULT_MODEL[ptype], {}, [d.id for d in docs]
    best = votes.most_common(1)[0][0]
    return best, params[best], [d.id for d in docs]


async def agents_predict(train: PreparedData, test: PreparedData, settings: Settings, retriever, *,
                         run_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Полный цикл Supervisor; тестовые строки подаются с пустой целью (как «новые студенты»)."""
    from backend.agents.base import AgentContext
    from backend.agents.supervisor import Supervisor
    from backend.core.llm.manager import build_llm_manager
    from backend.core.models import ForecastState
    from backend.data.cache import MemoryCache
    from backend.data.storage import ArtifactStore
    from backend.utils.tracing import MemoryRecorder, Tracer

    task_id = f"eval_{run_id}_{uuid.uuid4().hex[:8]}"
    store = ArtifactStore()
    df = pd.concat([train.X.assign(__y=train.y.values, __rid=[f"tr{i}" for i in range(len(train.X))]),
                    test.X.assign(__y=np.nan, __rid=[f"te{i}" for i in range(len(test.X))])], ignore_index=True)
    if train.groups is not None and test.groups is not None:
        df[STUDENT_GROUP] = pd.concat([train.groups, test.groups], ignore_index=True)
    path = store.path(task_id, "input.parquet")
    df.to_parquet(path, index=False)
    rec = MemoryRecorder()
    ctx = AgentContext(settings=settings, llm=build_llm_manager(settings, rec, MemoryCache(), task_id),
                       tracer=Tracer(task_id, rec), registry=default_registry(), store=store, retriever=retriever)
    state = ForecastState(task_id=task_id, series_id=run_id, data_path=str(path), target_column="__y",
                          id_column="__rid", problem_type=train.problem_type, max_iterations=settings.max_iterations,
                          metadata={"explain": False})
    t0 = time.perf_counter()
    state = await Supervisor().run(state, ctx)
    hitl = state.hitl_requested
    if hitl:  # в оценке решение «человека» — принять текущий результат
        state = await Supervisor().resume(state, ctx, {"action": "accept"})
    elapsed = time.perf_counter() - t0
    pred = pd.read_csv(state.predictions_path)
    pred = pred[pred["split"] == "new"].set_index("id")
    col = "risk_probability" if train.problem_type == "classification" else "predicted_score"
    p = pred.loc[[f"te{i}" for i in range(len(test.X))], col].values
    llm_models = sorted({f"{call['provider']}:{call['model_name']}" for call in rec.llm_calls if call["status"] == "ok"})
    agent = {"iterations": state.iteration, "hitl": hitl, "time_s": round(elapsed, 3), "tokens": rec_tokens(rec),
             "llm_calls": len(rec.llm_calls), "model": state.model_type, "selection_mode": state.selection_mode,
             "llm_models": llm_models,
             "editor_actions": len(state.editor_actions), "critic_score": state.critic_score,
             "threshold": state.decision_threshold}
    return p, agent


def rec_tokens(rec) -> int:
    return int(sum(c.get("tokens_used", 0) or 0 for c in rec.llm_calls))


def foundation_predict(train: PreparedData, test: PreparedData) -> np.ndarray | None:
    """+foundation: TabPFN (табличная foundation-модель), если установлен."""
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore
    except ImportError:
        return None
    from backend.tabular.preprocessing import build_preprocessor  # pragma: no cover

    prep = build_preprocessor(train.numeric_cols, train.categorical_cols)  # pragma: no cover
    Xtr, Xte = prep.fit_transform(train.X), prep.transform(test.X)  # pragma: no cover
    if train.problem_type == "classification":  # pragma: no cover
        return TabPFNClassifier().fit(Xtr, train.y).predict_proba(Xte)[:, 1]
    return TabPFNRegressor().fit(Xtr, train.y).predict(Xte)  # pragma: no cover


async def evaluate_spec(spec: DatasetSpec, *, modes: list[str], fractions: list[float], holdouts: list[float],
                        retriever: HybridRetriever | None, settings: Settings | None = None,
                        uci_cache: dict | None = None) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    registry = default_registry()
    df = build_dataset(spec, uci_cache)
    data = prepare_dataframe(df, **prepare_kwargs(spec))
    rows: list[dict[str, Any]] = []
    for holdout in holdouts:
        train_full, test = split_holdout(data, holdout, settings.random_seed)
        for frac in fractions:
            train = subsample(train_full, frac, settings.random_seed)
            if data.problem_type == "classification" and train.y.value_counts().min() < 3:
                log.warning("%s frac=%s: слишком мало примеров класса — пропуск", spec.id, frac)
                continue
            base = {"dataset": spec.id, "source": spec.source, "problem_type": data.problem_type,
                    "holdout": holdout, "fraction": frac, "n_train": len(train.X), "n_test": len(test.X),
                    "n_features": data.X.shape[1],
                    "n_test_minority": int(min(test.y.sum(), len(test.y) - test.y.sum()))
                    if data.problem_type == "classification" else None}
            for mode in modes:
                t0 = time.perf_counter()
                agent: dict[str, Any] = {}
                model = None
                try:
                    if mode == "baseline":
                        model = DEFAULT_MODEL[data.problem_type]
                        pred = await asyncio.to_thread(fit_predict, registry, model, {}, train, test, settings.random_seed)
                    elif mode == "+rag":
                        model, args, _ = rag_choice(retriever, train, settings.rag_top_k)
                        pred = await asyncio.to_thread(fit_predict, registry, model, args, train, test, settings.random_seed)
                    elif mode == "+agents":
                        pred, agent = await agents_predict(train, test, settings, retriever, run_id=spec.id)
                        model = agent["model"]
                    elif mode == "+foundation":
                        pred = await asyncio.to_thread(foundation_predict, train, test)
                        if pred is None:
                            continue
                        model = "tabpfn"
                    else:
                        raise ValueError(mode)
                except Exception as e:
                    log.error("%s %s failed: %s", spec.id, mode, e)
                    rows.append({**base, "mode": mode, "error": str(e)})
                    continue
                metrics = clean(compute_metrics(data.problem_type, test.y.values, pred))
                rows.append({**base, "mode": mode, "model": model, "time_s": round(time.perf_counter() - t0, 3),
                             **{f"m_{k}": v for k, v in metrics.items()}, **{f"a_{k}": v for k, v in agent.items()},
                             "_loss": per_sample_loss(data.problem_type, test.y.values, pred).tolist()})
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame([r for r in rows if "error" not in r])
    out: dict[str, Any] = {"groups": [], "tests": [], "dm": [], "agents": {},
                           "successful_runs": len(df), "errors": len(rows) - len(df)}
    if df.empty:
        return out
    for ptype, g in df.groupby("problem_type"):
        pm = f"m_{primary_metric(ptype)}"
        metric_cols = [c for c in g.columns if c.startswith("m_")]
        agg = g.groupby(["mode", "holdout", "fraction"])[metric_cols + ["time_s"]].mean().reset_index()
        for rec in agg.to_dict("records"):
            out["groups"].append(clean({"problem_type": ptype, **rec}))
        for (holdout, frac), gg in g.groupby(["holdout", "fraction"]):
            piv = gg.pivot_table(index="dataset", columns="mode", values=pm, aggfunc="mean")
            for a, b in [("+rag", "baseline"), ("+agents", "baseline"), ("+agents", "+rag"),
                         ("+foundation", "baseline")]:
                if a in piv.columns and b in piv.columns:
                    pair = piv[[a, b]].dropna()
                    if len(pair) >= 2:
                        out["tests"].append(clean({"problem_type": ptype, "holdout": holdout, "fraction": frac,
                                                   "metric": pm[2:], **compare_paired(pair[a], pair[b], a, b)}))
                    # DM-тест по каждому датасету на поэлементных потерях
                    sig, better, n = 0, 0, 0
                    for _, dg in gg.groupby("dataset"):
                        la = dg.loc[dg["mode"] == a, "_loss"]
                        lb = dg.loc[dg["mode"] == b, "_loss"]
                        if len(la) and len(lb):
                            res = diebold_mariano(la.iloc[0], lb.iloc[0])
                            n += 1
                            if res["pvalue"] < 0.05:
                                sig += 1
                                better += res["dm"] < 0
                    if n:
                        out["dm"].append({"problem_type": ptype, "holdout": holdout, "fraction": frac, "a": a, "b": b,
                                          "datasets": n, "significant": sig, "a_better_significant": int(better)})
    ag = df[df["mode"] == "+agents"]
    if len(ag):
        out["agents"] = clean({
            "runs": int(len(ag)), "avg_iterations": float(ag["a_iterations"].mean()),
            "hitl_rate": float(ag["a_hitl"].astype(float).mean()), "avg_time_s": float(ag["a_time_s"].mean()),
            "avg_tokens": float(ag["a_tokens"].mean()), "avg_llm_calls": float(ag["a_llm_calls"].mean()),
            "selection_modes": ag["a_selection_mode"].value_counts().to_dict(),
            "models": ag["model"].value_counts().to_dict(),
            "llm_models_by_run": ag["a_llm_models"].map(lambda models: ", ".join(models) or "нет").value_counts().to_dict(),
        })
    return out


def compare_models(data: PreparedData, models: list[str], sample_size: int | None = None, folds: int = 5,
                   seed: int = 42) -> dict[str, Any]:
    """CV-сравнение моделей (дефолтные гиперпараметры) + тесты значимости по фолдам и DM на OOF-потерях."""
    registry = default_registry()
    if sample_size and sample_size < len(data.X):
        data = subsample(data, sample_size / len(data.X), seed)
    ptype = data.problem_type
    pm = primary_metric(ptype)
    fold_scores: dict[str, list[float]] = {m: [] for m in models}
    oof: dict[str, np.ndarray] = {m: np.zeros(len(data.X)) for m in models}
    for tr, te in cv_splits(data, folds, seed):
        trd, ted = subset(data, tr), subset(data, te)
        for m in models:
            p = fit_predict(registry, m, {}, trd, ted, seed)
            oof[m][te] = p
            fold_scores[m].append(compute_metrics(ptype, ted.y.values, p)[pm])
    results = {m: clean({**compute_metrics(ptype, data.y.values, oof[m]), "fold_scores": fold_scores[m],
                         "fold_mean": float(np.mean(fold_scores[m])), "fold_std": float(np.std(fold_scores[m]))})
               for m in models}
    ranking = sorted(models, key=lambda m: -results[m]["fold_mean"])
    tests = []
    for other in ranking[1:]:
        t = compare_paired(fold_scores[ranking[0]], fold_scores[other], ranking[0], other)
        t["diebold_mariano"] = diebold_mariano(per_sample_loss(ptype, data.y, oof[ranking[0]]),
                                               per_sample_loss(ptype, data.y, oof[other]))
        tests.append(clean(t))
    return {"problem_type": ptype, "metric": pm, "sample_size": int(len(data.X)), "results": results,
            "ranking": ranking, "winner": ranking[0], "tests": tests}
