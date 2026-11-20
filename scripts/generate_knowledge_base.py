"""Генерация базы знаний.

1) student_case — профили студентов (UCI 1044 + синтетические) → итог + рекомендации;
2) dataset_case — для каждого набора (UCI-подвыборки и синтетика, split=kb) прогоняется сетка
   классических моделей (3-fold CV), лучшая сохраняется как «успешный кейс»;
3) eval_specs.json — отдельный тестовый набор спецификаций (split=eval, seed-ы не пересекаются).

python scripts/generate_knowledge_base.py --datasets 250 --synthetic-students 1500 --n-jobs 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402

from backend.core.config import get_settings  # noqa: E402
from backend.core.llm.tools import default_registry  # noqa: E402
from backend.core.metrics import clean, compute_metrics, primary_metric  # noqa: E402
from backend.data.embeddings import TabularFeatureExtractor  # noqa: E402
from backend.data.knowledge_base import (  # noqa: E402
    canonical_profile,
    describe_dataset_case,
    describe_student,
    recommendations_for,
)
from backend.evaluation.specs import build_dataset, make_specs, prepare_kwargs  # noqa: E402
from backend.tabular.datasets import generate_synthetic, load_uci  # noqa: E402
from backend.tabular.preprocessing import prepare_dataframe  # noqa: E402
from backend.tabular.training import cv_splits, make_pipeline  # noqa: E402

GRID = {
    "classification": [
        ("fit_logistic_regression", {"C": 1.0}),
        ("fit_logistic_regression", {"C": 0.1, "l1_ratio": 0.5}),
        ("fit_random_forest", {"n_estimators": 200, "min_samples_leaf": 2}),
        ("fit_random_forest", {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5, "class_weight": "balanced"}),
        ("fit_tabular_xgboost", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}),
        ("fit_tabular_xgboost", {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1}),
    ],
    "regression": [
        ("fit_ridge_regression", {"alpha": 1.0}),
        ("fit_ridge_regression", {"alpha": 20.0}),
        ("fit_random_forest", {"n_estimators": 200, "min_samples_leaf": 2}),
        ("fit_random_forest", {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5}),
        ("fit_tabular_xgboost", {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05}),
        ("fit_tabular_xgboost", {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1}),
    ],
}


def student_cases(n_synthetic: int, seed: int = 0) -> list[dict]:
    cases = []
    uci = load_uci("both")
    for row in uci.to_dict("records"):
        g3 = float(row["G3"])
        status = "fail" if g3 < 10 else ("excellent" if g3 >= 15 else "pass")
        prof = canonical_profile(row)
        at_risk = status == "fail"
        outcome_txt = {"fail": "не сдал курс", "pass": "сдал курс", "excellent": "сдал на отлично"}[status]
        cases.append({
            "source_id": f"student_{row['student_id']}", "kind": "student_case", "task_type": "tabular",
            "problem_type": "classification", "target_column": "G3", "series_features": prof,
            "best_model": None, "hyperparams": {}, "metrics": {"G3": g3},
            "description": f"{describe_student(prof, row)} Итог: G3={g3:.0f}/20, {outcome_txt}.",
            "meta": {"source": f"uci_{row['subject']}", "status": "at_risk" if at_risk else status, "outcome": g3,
                     "recommendations": recommendations_for(prof, at_risk)},
        })
    per = max(1, n_synthetic // 3)
    for k in range(3):
        df = generate_synthetic(per, seed=seed + 500 + k, missing_rate=0.0)
        for row in df.to_dict("records"):
            prof = canonical_profile(row)
            dropped = int(row["dropout"]) == 1
            status = "dropout" if dropped else ("fail" if row["final_score"] < 40 else "pass")
            at_risk = status != "pass"
            cases.append({
                "source_id": f"student_{row['student_id']}", "kind": "student_case", "task_type": "tabular",
                "problem_type": "classification", "target_column": "dropout", "series_features": prof,
                "best_model": None, "hyperparams": {}, "metrics": {"final_score": float(row["final_score"])},
                "description": f"{describe_student(prof, row)} Итог: "
                               f"{'отчислен' if dropped else 'завершил курс'}, итоговый балл {row['final_score']:.0f}/100.",
                "meta": {"source": "synthetic_oulad", "status": "at_risk" if at_risk else "ok",
                         "outcome": float(row["final_score"]), "recommendations": recommendations_for(prof, at_risk)},
            })
    return cases


def _no_parallel(est):
    if "n_jobs" in est.get_params():
        est.set_params(n_jobs=1)
    return est


def dataset_case(spec_dict: dict, seed: int = 42) -> dict | None:
    from backend.evaluation.specs import DatasetSpec

    spec = DatasetSpec(**spec_dict)
    reg = default_registry()
    try:
        df = build_dataset(spec)
        data = prepare_dataframe(df, **prepare_kwargs(spec))
    except Exception as e:  # pragma: no cover
        return {"error": str(e), "id": spec.id}
    ptype = data.problem_type
    if ptype == "classification" and data.y.value_counts().min() < 6:
        return None
    meta = TabularFeatureExtractor().extract(X=data.X, y=data.y, problem_type=ptype,
                                             numeric_cols=data.numeric_cols, categorical_cols=data.categorical_cols)
    pm = primary_metric(ptype)
    cv = list(cv_splits(data, 3, seed))
    scores = []
    for name, args in GRID[ptype]:
        tool = reg.get(name)
        a = tool.validate_args(args)
        pipe = make_pipeline(_no_parallel(tool.builder(a, ptype, seed)), data, scale=tool.scale)
        method = "predict_proba" if ptype == "classification" else "predict"
        oof = cross_val_predict(pipe, data.X, data.y, cv=cv, method=method)
        if ptype == "classification":
            oof = oof[:, 1]
        m = clean(compute_metrics(ptype, data.y, oof))
        scores.append((name, a, m))
    best_name, best_args, best_m = max(scores, key=lambda s: s[2].get(pm) or -1e9)
    return {
        "source_id": f"dataset_{spec.id}", "kind": "dataset_case", "task_type": "tabular", "problem_type": ptype,
        "target_column": spec.target, "series_features": meta, "best_model": best_name, "hyperparams": best_args,
        "metrics": best_m, "description": describe_dataset_case(meta, best_name, best_m, spec.source),
        "meta": {"source": spec.source, "spec": spec.to_dict(),
                 "candidates": [{"model": n, "params": a, pm: m.get(pm)} for n, a, m in scores]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=int, default=250)
    ap.add_argument("--synthetic-students", type=int, default=1500)
    ap.add_argument("--eval-specs", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = get_settings().kb_dir
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    students = student_cases(a.synthetic_students, a.seed)
    with open(out / "student_cases.jsonl", "w", encoding="utf-8") as f:
        for c in students:
            f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")
    print(f"student_cases: {len(students)} ({time.time() - t0:.1f}s)")

    specs = make_specs(a.datasets, "kb", a.seed)
    t0 = time.time()
    results = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(dataset_case)(s.to_dict()) for s in specs)
    cases = [r for r in results if r and "error" not in r]
    with open(out / "dataset_cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")
    wins: dict[str, int] = {}
    for c in cases:
        wins[c["best_model"]] = wins.get(c["best_model"], 0) + 1
    print(f"dataset_cases: {len(cases)} ({time.time() - t0:.1f}s); лучшие модели: {wins}")

    eval_specs = [s.to_dict() for s in make_specs(a.eval_specs, "eval", a.seed)]
    (out / "eval_specs.json").write_text(json.dumps(eval_specs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"eval_specs: {len(eval_specs)} (split=eval, seed-ы не пересекаются с kb)")
    print(f"ИТОГО записей базы знаний: {len(students) + len(cases)}")


if __name__ == "__main__":
    main()
