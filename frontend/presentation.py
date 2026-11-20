"""Formatting and charts for the student forecast workspace."""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODELS = {
    "fit_logistic_regression": "Логистическая регрессия", "fit_random_forest": "Random Forest",
    "fit_tabular_xgboost": "XGBoost", "fit_ridge_regression": "Ridge",
}
STATUSES = {
    "pending": "В очереди", "running": "Идёт расчёт", "completed": "Прогноз готов",
    "waiting_for_human": "Нужно решение преподавателя", "failed": "Не удалось завершить",
}
METRICS = {"roc_auc": "ROC-AUC", "r2": "R²", "f1": "F1", "mae": "MAE", "rmse": "RMSE",
           "accuracy": "Точность", "precision": "Precision", "recall": "Полнота", "brier": "Brier", "ece": "ECE"}
LEVELS = {"high": "Высокий", "medium": "Средний", "low": "Низкий"}
MODE_LABELS = {"baseline": "Базовый", "+rag": "С поиском кейсов", "+agents": "С агентами"}
PALETTE = ["#087f78", "#3971cb", "#b07b21", "#8664c8", "#cd6078"]


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def banner(title, detail="", kind="info"):
    return (f'<div class="notice notice-{kind}" role="status"><span class="notice-dot"></span>'
            f'<div><strong>{escape(str(title))}</strong>'
            + (f'<p>{escape(str(detail))}</p>' if detail else "") + "</div></div>")


def summary_cards(result):
    problem = result.get("problem_type")
    primary = "r2" if problem == "regression" else "roc_auc"
    test = (result.get("metrics") or {}).get("test") or {}
    score = result.get("critic_score")
    cards = [
        ("Модель прогноза", MODELS.get(result.get("model_type"), result.get("model_type") or "—"), ""),
        (METRICS[primary], fmt(test.get(primary)), "на отложенной выборке"),
        ("Оценка критика", f"{fmt(score, 1)} / 10" if score is not None else "—", "проверка агентом"),
    ]
    return '<div class="metric-grid">' + "".join(
        f'<div class="metric-card"><span>{escape(label)}</span><strong>{escape(value)}</strong>'
        f'<small>{escape(note)}</small></div>' for label, value, note in cards
    ) + "</div>"


def metrics_table(result):
    metrics = result.get("metrics") or {}
    cv, test = metrics.get("cv") or {}, metrics.get("test") or {}
    return pd.DataFrame([(METRICS.get(key, key), fmt(cv.get(key)), fmt(test.get(key)))
                         for key in dict.fromkeys([*cv, *test])],
                        columns=["Метрика", "Кросс-валидация", "Отложенная выборка"])


def predictions_table(result):
    frame = pd.DataFrame(result.get("predictions_preview") or [])
    columns = {
        "id": "Студент", "risk_probability": "Вероятность риска", "risk_level": "Уровень риска",
        "predicted_score": "Прогноз оценки", "split": "Тип записи", "recommendations": "Рекомендации",
    }
    existing = [key for key in columns if key in frame]
    if "risk_probability" in frame:
        frame["risk_probability"] = frame["risk_probability"].map(lambda value: fmt(value * 100, 1) + " %"
                                                                  if isinstance(value, (float, int)) else value)
    if "risk_level" in frame:
        frame["risk_level"] = frame["risk_level"].replace(LEVELS)
    if "split" in frame:
        frame["split"] = frame["split"].replace({"new": "Новый студент", "labeled": "Известен результат"})
    if "recommendations" in frame:
        frame["recommendations"] = frame["recommendations"].str.split(" | ", n=1, regex=False).str[0]
    return frame[existing].rename(columns=columns)


def chart(fig, *, height=310):
    fig.update_layout(
        height=height, margin=dict(l=12, r=18, t=22, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE, font=dict(family="Segoe UI, sans-serif", color="#74879c", size=13),
        legend=dict(orientation="h", title_text=""),
    )
    fig.update_xaxes(gridcolor="rgba(126,146,169,.19)", automargin=True)
    fig.update_yaxes(gridcolor="rgba(126,146,169,.19)", automargin=True)
    return fig


def importance_plot(imp):
    if not imp:
        return None
    items = sorted(imp.items(), key=lambda item: item[1], reverse=True)[:10][::-1]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=PALETTE[0]))
    fig.update_xaxes(title_text="Влияние на качество прогноза")
    return chart(fig, height=344)


def risk_plot(counts):
    if not counts:
        return None
    levels = [key for key in LEVELS if key in counts]
    fig = go.Figure(go.Bar(x=[LEVELS[key] for key in levels], y=[counts[key] for key in levels],
                           marker_color=[{"high": "#ce5971", "medium": "#c49137", "low": "#359387"}[key]
                                         for key in levels],
                           text=[counts[key] for key in levels], textposition="outside", cliponaxis=False))
    fig.update_yaxes(title_text="Студенты", rangemode="tozero")
    return chart(fig, height=344)


def batch_plot(groups):
    specs = [(metric, title) for metric, title in (("m_roc_auc", "ROC-AUC"), ("m_r2", "R²"))
             if metric in groups and groups[metric].notna().any()]
    if not specs:
        return None
    fig = make_subplots(rows=len(specs), cols=1, shared_xaxes=True, vertical_spacing=0.18)
    for row, (metric, title) in enumerate(specs, 1):
        for index, (mode, records) in enumerate(groups.dropna(subset=[metric]).groupby("mode")):
            records = records.sort_values("fraction")
            fig.add_scatter(x=records["fraction"], y=records[metric], mode="lines+markers",
                            name=MODE_LABELS.get(mode, mode), legendgroup=mode, showlegend=row == 1,
                            line_color=PALETTE[index], row=row, col=1)
        fig.update_yaxes(title_text=title, row=row, col=1)
    fig.update_xaxes(tickformat=".0%", title_text="Доля обучающих данных", row=len(specs), col=1)
    return chart(fig, height=260 * len(specs))


def tree_markdown(nodes, depth=0):
    lines = []
    for node in nodes:
        tool = f" · `{node['tool_name']}`" if node.get("tool_name") else ""
        indent = "&nbsp;" * 4 * depth
        icon = "✓" if node["status"] == "ok" else "•"
        lines.append(f"{indent}{icon} **{node['agent_name']}** · {node['step']}{tool} · {node['duration_ms']:.0f} мс")
        if node.get("output_summary"):
            lines.append(f"{indent}&nbsp;&nbsp;↳ {escape(node['output_summary'][:180])}")
        lines.extend(tree_markdown(node.get("children", []), depth + 1))
    return lines


def timeline(nodes):
    rows = []

    def walk(items, depth=0):
        for node in items:
            if node.get("started_at") and node.get("finished_at"):
                rows.append({"Шаг": f"{len(rows) + 1}. {'  ' * depth}{node['agent_name']} · {node['step']}",
                             "Начало": node["started_at"], "Конец": node["finished_at"], "Агент": node["agent_name"]})
            walk(node.get("children", []), depth + 1)

    walk(nodes)
    if not rows:
        return None
    fig = px.timeline(pd.DataFrame(rows), x_start="Начало", x_end="Конец", y="Шаг", color="Агент",
                      color_discrete_sequence=PALETTE)
    fig.update_yaxes(autorange="reversed")
    return chart(fig, height=max(260, 24 * len(rows) + 64))
