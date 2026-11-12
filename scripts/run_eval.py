"""Оценка режимов baseline / +rag / +agents / +foundation на отложенном наборе спецификаций.

Примеры:
  python scripts/run_eval.py --num-series 20 --modes baseline +rag +agents --data-fractions 0.1 0.2 0.5 1.0
  python scripts/run_eval.py --num-series 50 --horizons 10 20 50 --modes baseline +rag

Для табличной задачи «горизонт» интерпретируется как размер отложенной выборки новых студентов
в процентах (--horizons 10 20 50 => holdout 10%, 20%, 50%).
Результаты: results/eval_<ts>/{rows.csv, summary.json, report.md}, лог — eval.log; запись в БД.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from backend.core.config import get_settings  # noqa: E402
from backend.evaluation.runner import MODES, evaluate_spec, summarize  # noqa: E402
from backend.evaluation.specs import DatasetSpec, make_specs  # noqa: E402
from backend.retrieval.index_manager import get_retriever  # noqa: E402

log = logging.getLogger("run_eval")


def check_data(spec: DatasetSpec, n_train: int, n_test: int, problem_type: str, minority: int) -> list[str]:
    """Проверка достаточности данных (аналог проверок MASE/сезонности для таблиц)."""
    warns = []
    if n_train < 50:
        warns.append(f"{spec.id}: n_train={n_train} < 50 — оценки нестабильны")
    if n_test < 20:
        warns.append(f"{spec.id}: n_test={n_test} < 20 — широкие доверительные интервалы")
    if problem_type == "classification" and minority < 10:
        warns.append(f"{spec.id}: в тесте < 10 объектов миноритарного класса — ROC-AUC ненадёжен")
    return warns


def load_specs(n: int, seed: int) -> list[DatasetSpec]:
    path = get_settings().kb_dir / "eval_specs.json"
    if path.exists():
        specs = [DatasetSpec(**d) for d in json.loads(path.read_text(encoding="utf-8"))]
        if len(specs) >= n:
            return specs[:n]
    return make_specs(n, "eval", seed)


def markdown(summary: dict, args) -> str:
    lines = [f"# Отчёт run_eval ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"Наборов: {args.num_series}; режимы: {', '.join(args.modes)}; доли данных: {args.data_fractions}; "
             f"holdout: {args.holdouts}", ""]
    g = pd.DataFrame(summary["groups"])
    if not g.empty:
        for ptype, gg in g.groupby("problem_type"):
            cols = ["mode", "holdout", "fraction"] + [c for c in (
                ["m_roc_auc", "m_f1", "m_accuracy", "m_recall", "m_brier"] if ptype == "classification"
                else ["m_r2", "m_mae", "m_rmse"]) if c in gg] + ["time_s"]
            lines += [f"## {ptype}", "", gg[cols].round(4).to_markdown(index=False), ""]
    if summary.get("tests"):
        t = pd.DataFrame([{"задача": x["problem_type"], "holdout": x["holdout"], "доля": x["fraction"],
                           "сравнение": f"{x['a']} vs {x['b']}", "метрика": x["metric"],
                           "Δ среднее": x["mean_improvement"],
                           "95% CI": f"[{x['ci_improvement']['low']:.4f}; {x['ci_improvement']['high']:.4f}]",
                           "Wilcoxon p": x["wilcoxon"]["pvalue"], "t-test p": x["ttest"]["pvalue"],
                           "win rate": x["win_rate"]} for x in summary["tests"]])
        lines += ["## Статистические тесты (парные по наборам)", "", t.round(4).to_markdown(index=False), ""]
    if summary.get("dm"):
        lines += ["## Тест Диболда–Мариано по наборам (поэлементные потери)", "",
                  pd.DataFrame(summary["dm"]).to_markdown(index=False), ""]
    if summary.get("agents"):
        lines += ["## Агентные метрики (+agents)", "", "```json",
                  json.dumps(summary["agents"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


async def main_async(args) -> Path:
    settings = get_settings()
    out = ROOT / "results" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(ROOT / "eval.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in ("run_eval", "mlf"):
        lg = logging.getLogger(name)
        lg.addHandler(fh)
        lg.setLevel(logging.INFO)
    retriever = get_retriever()
    log.info("start: %s; RAG docs=%d; LLM=%s", vars(args), retriever.size, settings.llm_provider)
    specs = load_specs(args.num_series, args.seed)
    rows, cache, data_warnings = [], {}, []
    t0 = time.time()
    for i, spec in enumerate(specs, 1):
        r = await evaluate_spec(spec, modes=args.modes, fractions=args.data_fractions, holdouts=args.holdouts,
                                retriever=retriever, settings=settings, uci_cache=cache)
        for row in r:
            if "error" in row:
                continue
            if row["mode"] == args.modes[0]:
                data_warnings += check_data(spec, row["n_train"], row["n_test"], row["problem_type"],
                                            row.get("n_test_minority") or 0)
        rows += r
        log.info("[%d/%d] %s (%s): %d строк результатов", i, len(specs), spec.id, spec.source, len(r))
        print(f"[{i}/{len(specs)}] {spec.id} {spec.source} -> {len(r)} rows ({time.time() - t0:.0f}s)", flush=True)
    for w in data_warnings:
        log.warning(w)
    summary = summarize(rows)
    summary["data_warnings"] = data_warnings
    summary["config"] = {**vars(args), "llm_provider": settings.llm_provider, "rag_docs": retriever.size,
                         "elapsed_s": round(time.time() - t0, 1)}
    pd.DataFrame([{k: v for k, v in r.items() if k != "_loss"} for r in rows]).to_csv(out / "rows.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(markdown(summary, args), encoding="utf-8")
    if args.save_db:
        await save_db(summary, rows, args)
    log.info("done in %.1fs -> %s", time.time() - t0, out)
    return out


async def save_db(summary: dict, rows: list[dict], args) -> None:
    from backend.db.models import EvaluationResult, ModelComparison
    from backend.db.session import dispose_engine, init_db, session_scope

    await init_db()
    run_id = time.strftime("%Y%m%d%H%M%S")
    async with session_scope() as s:
        for g in summary["groups"]:
            s.add(EvaluationResult(run_id=run_id, model_name=g["mode"], mode=g["mode"],
                                   sample_size=args.num_series, metrics_json=g,
                                   evaluation_time=summary["config"]["elapsed_s"]))
        for t in summary["tests"]:
            winner = t["a"] if t["mean_improvement"] > 0 else t["b"]
            s.add(ModelComparison(models_compared=[t["a"], t["b"]], sample_size=args.num_series, results_json=t,
                                  winner=winner))
    await dispose_engine()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-series", type=int, default=20, help="число наборов данных для оценки (20/50/100)")
    ap.add_argument("--horizons", type=int, nargs="*", default=None,
                    help="размеры отложенной выборки в %% (для таблиц — аналог горизонта)")
    ap.add_argument("--holdouts", type=float, nargs="*", default=None, help="доли holdout (альтернатива --horizons)")
    ap.add_argument("--data-fractions", type=float, nargs="*", default=[1.0])
    ap.add_argument("--modes", nargs="*", default=["baseline", "+rag", "+agents"], choices=MODES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-db", dest="save_db", action="store_false")
    a = ap.parse_args(argv)
    if a.holdouts is None:
        a.holdouts = [h / 100 for h in a.horizons] if a.horizons else [0.2]
    return a


if __name__ == "__main__":
    out_dir = asyncio.run(main_async(parse_args()))
    print("Результаты:", out_dir)
