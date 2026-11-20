"""Графики для отчёта по последнему (или указанному) прогону run_eval -> reports/figures/*.png.

python scripts/make_report_plots.py [results/eval_YYYYmmdd_HHMMSS] [префикс_файлов]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"baseline": "#7f7f7f", "+rag": "#1f77b4", "+agents": "#d62728", "+foundation": "#2ca02c"}
PRIMARY = {"classification": "m_roc_auc", "regression": "m_r2"}
METRIC_LABELS = {"classification": "ROC-AUC", "regression": "R²"}


def latest_run() -> Path:
    runs = sorted((ROOT / "results").glob("eval_*"))
    if not runs:
        raise SystemExit("нет результатов: запустите scripts/run_eval.py")
    return runs[-1]


def main(run: Path, prefix: str = "") -> list[Path]:
    rows = pd.read_csv(run / "rows.csv")
    rows = rows[rows.get("error").isna()] if "error" in rows else rows
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. качество vs доля данных
    ptypes = sorted(rows["problem_type"].unique())
    fig, axes = plt.subplots(1, len(ptypes), figsize=(6 * len(ptypes), 4), squeeze=False)
    for ax, pt in zip(axes[0], ptypes, strict=True):
        m = PRIMARY[pt]
        g = rows[rows.problem_type == pt].groupby(["mode", "fraction"])[m].agg(["mean", "sem"]).reset_index()
        for mode, gg in g.groupby("mode"):
            ax.errorbar(gg["fraction"], gg["mean"], yerr=gg["sem"], marker="o", capsize=3, label=mode,
                        color=COLORS.get(mode))
        ax.set_xscale("log")
        ax.set_xticks(sorted(rows.fraction.unique()))
        ax.set_xticklabels([f"{int(f * 100)}%" for f in sorted(rows.fraction.unique())])
        ax.set_title(f"{METRIC_LABELS[pt]} при разном объёме обучения")
        ax.set_xlabel("доля обучающей выборки")
        ax.set_ylabel(METRIC_LABELS[pt])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    saved.append(out / f"{prefix}quality_vs_fraction.png")
    fig.savefig(saved[-1], dpi=130)
    plt.close(fig)

    # 2. стоимость режимов
    cost = rows.groupby("mode")["time_s"].mean().reindex([m for m in COLORS if m in rows["mode"].unique()])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cost.index, cost.values, color=[COLORS[m] for m in cost.index])
    ax.set_yscale("log")
    ax.set_ylabel("среднее время на набор, с (log)")
    ax.set_title("Стоимость режимов (время)")
    for i, v in enumerate(cost.values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    saved.append(out / f"{prefix}cost_time.png")
    fig.savefig(saved[-1], dpi=130)
    plt.close(fig)

    # 3. агентные метрики
    ag = rows[rows["mode"] == "+agents"]
    if len(ag):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ag["model"].value_counts().plot.bar(ax=axes[0], color="#d62728")
        axes[0].set_title("Модели, выбранные агентами")
        ag.groupby("fraction")["a_tokens"].mean().plot.bar(ax=axes[1], color="#9467bd")
        axes[1].set_title("Среднее число LLM-токенов на задачу")
        ag.groupby("fraction")["a_iterations"].mean().plot.bar(ax=axes[2], color="#8c564b")
        axes[2].set_title("Среднее число итераций Supervisor")
        for a in axes:
            a.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        saved.append(out / f"{prefix}agent_metrics.png")
        fig.savefig(saved[-1], dpi=130)
        plt.close(fig)

    # 4. приросты относительно baseline с 95% CI
    tests = pd.DataFrame(summary.get("tests", []))
    if len(tests):
        tests = tests[tests["b"] == "baseline"]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.27 * len(tests) + 1)))
        labels = [f"{METRIC_LABELS[r.problem_type]} {r.a} · обучение {int(r.fraction * 100)}% · тест {int(r.holdout * 100)}%"
                  for r in tests.itertuples()]
        lo = [r["ci_improvement"]["low"] for r in tests.to_dict("records")]
        hi = [r["ci_improvement"]["high"] for r in tests.to_dict("records")]
        mean = tests["mean_improvement"].values
        ax.errorbar(mean, range(len(mean)), xerr=[mean - lo, hi - mean], fmt="o", capsize=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("прирост основной метрики относительно baseline (bootstrap 95% CI)")
        fig.tight_layout()
        saved.append(out / f"{prefix}improvement_ci.png")
        fig.savefig(saved[-1], dpi=130)
        plt.close(fig)
    return saved


if __name__ == "__main__":
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    for p in main(run, sys.argv[2] if len(sys.argv) > 2 else ""):
        print(p)
