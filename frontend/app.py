"""Gradio UI: вход, прогноз, HITL, оценка качества, дерево рассуждений агентов (обновление в реальном времени).

python -m frontend.app   (переменная MLF_API_URL — адрес API, по умолчанию http://127.0.0.1:8000)
"""
from __future__ import annotations

import json
import os

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from frontend.api_client import ApiClient, ApiError

MODELS_CLF = ["fit_logistic_regression", "fit_random_forest", "fit_tabular_xgboost"]
ICONS = {"ok": "✅", "error": "❌"}
LEVEL_COLORS = {"high": "#d1495b", "medium": "#edae49", "low": "#00798c"}


def client(token: str | None) -> ApiClient:
    return ApiClient(token=token or None)


def err(e: Exception) -> str:
    return f"⚠️ {e}"


# ------------------------------------------------------------------ вход
def do_login(username, password):
    try:
        c = ApiClient()
        data = c.login(username, password)
        return data["access_token"], f"Вход выполнен: **{username}** (роль `{data['role']}`)"
    except ApiError as e:
        return None, err(e)


def do_register(email, username, password):
    try:
        u = ApiClient().register(email, username, password)
        return f"Зарегистрирован **{u['username']}** (роль `{u['role']}`). Теперь войдите."
    except ApiError as e:
        return err(e)


# ------------------------------------------------------------------ прогноз
def start_forecast(token, file, dataset, target, problem_type, threshold, max_iter):
    try:
        params = {"target_column": target or None, "problem_type": problem_type,
                  "binarize_threshold": threshold if threshold not in (None, 0) else None,
                  "max_iterations": int(max_iter) if max_iter else None}
        if file is None:
            params["dataset"] = dataset
        res = client(token).create_forecast(file if file else None, **params)
        return res["task_id"], f"Задача **{res['task_id']}** принята (HTTP 202, исполнение: {res['execution']})"
    except ApiError as e:
        return "", err(e)


def importance_plot(imp: dict | None):
    if not imp:
        return None
    items = list(imp.items())[:15][::-1]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h"))
    fig.update_layout(title="Важность признаков (permutation)", height=420, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def risk_plot(counts: dict | None):
    if not counts:
        return None
    df = pd.DataFrame({"уровень": list(counts), "студентов": list(counts.values())})
    return px.bar(df, x="уровень", y="студентов", color="уровень", color_discrete_map=LEVEL_COLORS,
                  title="Распределение по уровню риска", height=320)


def refresh_result(token, task_id):
    empty = ("", None, None, None, None, None)
    if not task_id:
        return ("Укажите task_id",) + empty[1:]
    try:
        c = client(token)
        st = c.status(task_id)
        head = f"**Статус:** `{st['status']}` · итерация {st.get('iteration')}"
        if st.get("hitl_reason"):
            head += f"\n\n🧑‍🏫 **Требуется решение человека:** {st['hitl_reason']}"
        if st.get("error"):
            head += f"\n\n```\n{st['error'][:800]}\n```"
        if st["status"] not in ("completed", "waiting_for_human"):
            return (head,) + empty[1:]
        r = c.result(task_id)
        cv = (r.get("metrics") or {}).get("cv", {})
        summary = (f"{head}\n\n**Модель:** `{r['model_type']}` (выбор: {r['selection_mode']}) · порог {r['decision_threshold']}"
                   f" · оценка критика **{r['critic_score']}**/10\n\n> {r.get('critic_reasoning') or ''}\n\n"
                   f"**Замечания:** " + (", ".join(f"{i['severity']}:{i['code']}" for i in r["issues"]) or "нет") +
                   "\n\n**Правки Editor:** " + (", ".join(a["tool"] for a in r["editor_actions"]) or "нет") +
                   f"\n\n**Токены LLM:** {r.get('tokens_used')}")
        metrics = pd.DataFrame([{"метрика": k, "CV": v, "holdout": (r["metrics"].get("test") or {}).get(k)}
                                for k, v in cv.items()])
        preds = pd.DataFrame(r.get("predictions_preview") or [])
        file = c.download_predictions(task_id) if r["status"] == "completed" else None
        return summary, metrics, importance_plot(r.get("feature_importance")), risk_plot(r.get("risk_counts")), preds, file
    except ApiError as e:
        return (err(e),) + empty[1:]


_finished: set[str] = set()


def auto_refresh(token, task_id, enabled):
    """Опрос статуса, пока задача не завершилась; завершённую задачу загружаем один раз."""
    if not enabled or not task_id or task_id in _finished:
        return (gr.skip(),) * 6
    out = refresh_result(token, task_id)
    if "`completed`" in out[0] or "`failed`" in out[0]:
        _finished.add(task_id)
    return out


# ------------------------------------------------------------------ HITL
def submit_task_hitl(token, task_id, action, model_type, threshold, comment):
    try:
        res = client(token).hitl(task_id, action=action, model_type=model_type if action == "choose_model" else None,
                                 threshold=threshold if action == "change_threshold" else None, comment=comment)
        return f"Решение отправлено, статус: {res['status']}"
    except ApiError as e:
        return err(e)


def load_queue(token, task_id):
    try:
        rows = client(token).queue(kind="student_risk", task_id=task_id or None)
        return pd.DataFrame([{"review_id": r["id"], "task": r["task_id"][:8], "студент": r["payload"].get("student_id"),
                              "риск": r["payload"].get("score"),
                              "факторы": "; ".join(d["feature"] for d in r["payload"].get("drivers", [])),
                              "рекомендации": " | ".join(r["payload"].get("recommendations", [])[:2])} for r in rows])
    except ApiError as e:
        return pd.DataFrame({"ошибка": [str(e)]})


def submit_review(token, review_id, decision, adjusted, comment):
    try:
        res = client(token).review(int(review_id), decision, adjusted if decision == "adjust" else None, comment)
        return f"Сохранено: {res['status']}; добавлено в базу знаний: {res['kb_added']}"
    except (ApiError, ValueError, TypeError) as e:
        return err(e)


# ------------------------------------------------------------------ оценка
def run_compare(token, models, sample_size):
    try:
        res = client(token).compare(models, int(sample_size) if sample_size else None)
        df = pd.DataFrame([{"модель": m, res["metric"]: r["fold_mean"], "std": r["fold_std"], "f1": r.get("f1"),
                            "recall": r.get("recall")} for m, r in res["results"].items()])
        fig = px.bar(df, x="модель", y=res["metric"], error_y="std", title=f"Сравнение моделей (winner: {res['winner']})")
        tests = pd.DataFrame([{"vs": f"{t['a']} vs {t['b']}", "Δ": t["mean_improvement"],
                               "Wilcoxon p": t["wilcoxon"]["pvalue"], "t-test p": t["ttest"]["pvalue"],
                               "DM p": t["diebold_mariano"]["pvalue"]} for t in res["tests"]])
        return df, fig, tests
    except ApiError as e:
        return pd.DataFrame({"ошибка": [str(e)]}), None, None


def start_batch(token, n, modes, fractions):
    try:
        fr = [float(x) for x in str(fractions).replace(" ", "").split(",") if x]
        return client(token).batch(num_datasets=int(n), modes=modes, fractions=fr)["task_id"]
    except (ApiError, ValueError) as e:
        return err(e)


def batch_status(token, job_id):
    try:
        res = client(token).batch_status(job_id)
        groups = pd.DataFrame((res.get("summary") or {}).get("groups", []))
        fig = None
        if not groups.empty:
            metric = "m_roc_auc" if "m_roc_auc" in groups and groups["m_roc_auc"].notna().any() else "m_r2"
            fig = px.line(groups.dropna(subset=[metric]), x="fraction", y=metric, color="mode", markers=True,
                          title="Качество vs доля обучающих данных")
        return f"Статус: {res['status']}", groups, fig
    except ApiError as e:
        return err(e), None, None


# ------------------------------------------------------------------ логи
def tree_markdown(nodes, depth=0) -> list[str]:
    lines = []
    for n in nodes:
        tool = f" · 🔧 `{n['tool_name']}`" if n.get("tool_name") else ""
        tok = f" · {n['tokens']} tok" if n.get("tokens") else ""
        lines.append(f"{'&nbsp;' * 4 * depth}{ICONS.get(n['status'], '•')} **{n['agent_name']}** / {n['step']}{tool}"
                     f" · {n['duration_ms']:.0f} ms{tok}")
        if n.get("output_summary"):
            lines.append(f"{'&nbsp;' * 4 * (depth + 1)}↳ `{n['output_summary'][:160]}`")
        lines += tree_markdown(n.get("children", []), depth + 1)
    return lines


def timeline(nodes):
    flat = []

    def walk(ns, depth):
        for n in ns:
            flat.append({"span": f"{'  ' * depth}{n['agent_name']}/{n['step']}", "start": n["started_at"],
                         "end": n["finished_at"], "agent": n["agent_name"]})
            walk(n.get("children", []), depth + 1)

    walk(nodes, 0)
    if not flat:
        return None
    df = pd.DataFrame(flat)
    fig = px.timeline(df, x_start="start", x_end="end", y="span", color="agent", title="Трассировка агентов")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(300, 22 * len(df)))
    return fig


def load_logs(token, task_id):
    if not task_id:
        return "Укажите task_id", None, None, None
    try:
        c = client(token)
        rs = c.reasoning(task_id)
        md = "\n\n".join(tree_markdown(rs["tree"])) or "Лог пуст (задача ещё не стартовала?)"
        calls = pd.DataFrame(rs["llm_calls"])
        return (f"Шагов: {rs['n_spans']} · токенов: {rs['total_tokens']}\n\n" + md, timeline(rs["tree"]), calls,
                json.dumps(c.agent_metrics(), ensure_ascii=False, indent=2))
    except ApiError as e:
        return err(e), None, None, None


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MLF · Прогноз успеваемости") as demo:
        token = gr.State(None)
        gr.Markdown("# 🎓 Прогноз успеваемости студентов\nМультиагентная система: DataAnalyst → ModelSelector "
                    "(Tool Calling + RAG) → Validator → Critic → Editor, Supervisor и Human-in-the-Loop.")
        with gr.Tab("Вход"):
            with gr.Row():
                with gr.Column():
                    u = gr.Textbox(label="Логин или email")
                    p = gr.Textbox(label="Пароль", type="password")
                    login_btn = gr.Button("Войти", variant="primary")
                    login_msg = gr.Markdown()
                    login_btn.click(do_login, [u, p], [token, login_msg])
                with gr.Column():
                    re_email = gr.Textbox(label="Email")
                    re_user = gr.Textbox(label="Имя пользователя")
                    re_pass = gr.Textbox(label="Пароль", type="password")
                    out = gr.Markdown()
                    gr.Button("Зарегистрироваться").click(do_register, [re_email, re_user, re_pass], out)
        with gr.Tab("Прогноз"):
            with gr.Row():
                with gr.Column(scale=1):
                    f = gr.File(label="CSV/Parquet со студентами (пустая цель = новые студенты)", type="filepath")
                    ds = gr.Dropdown(["uci_both", "uci_mat", "uci_por", "synthetic"], value="uci_both",
                                     label="…или встроенный датасет")
                    target = gr.Textbox(label="Целевая колонка", placeholder="G3 / dropout / final_score")
                    ptype = gr.Radio(["auto", "classification", "regression"], value="auto", label="Тип задачи")
                    thr = gr.Number(value=10, label="Порог бинаризации (цель < порога = риск); 0 — нет")
                    iters = gr.Slider(1, 5, value=3, step=1, label="max_iterations")
                    task_id = gr.Textbox(label="task_id")
                    msg = gr.Markdown()
                    gr.Button("Запустить", variant="primary").click(
                        start_forecast, [token, f, ds, target, ptype, thr, iters], [task_id, msg])
                    auto = gr.Checkbox(value=True, label="Автообновление (3 с)")
                    refresh = gr.Button("Обновить")
                with gr.Column(scale=2):
                    summary = gr.Markdown()
                    metrics = gr.Dataframe(label="Метрики")
                    with gr.Row():
                        imp = gr.Plot()
                        risk = gr.Plot()
                    preds = gr.Dataframe(label="Студенты с наибольшим риском", wrap=True)
                    dl = gr.File(label="Скачать прогноз")
            outs = [summary, metrics, imp, risk, preds, dl]
            refresh.click(refresh_result, [token, task_id], outs)
            timer = gr.Timer(3.0)
            timer.tick(auto_refresh, [token, task_id, auto], outs)
        with gr.Tab("HITL"):
            gr.Markdown("### Решение по задаче (модель не сошлась / критические замечания)")
            with gr.Row():
                h_task = gr.Textbox(label="task_id")
                h_action = gr.Dropdown(["accept", "choose_model", "change_threshold", "reject"], value="accept",
                                       label="Действие")
                h_model = gr.Dropdown(MODELS_CLF + ["fit_ridge_regression"], label="Модель")
                h_thr = gr.Number(value=0.5, label="Порог")
            h_comment = gr.Textbox(label="Комментарий")
            h_out = gr.Markdown()
            gr.Button("Отправить решение").click(submit_task_hitl, [token, h_task, h_action, h_model, h_thr, h_comment],
                                                 h_out)
            gr.Markdown("### Очередь студентов высокого риска (роль editor/admin)")
            q_task = gr.Textbox(label="Фильтр по task_id (необязательно)")
            q_table = gr.Dataframe(wrap=True)
            gr.Button("Загрузить очередь").click(load_queue, [token, q_task], q_table)
            with gr.Row():
                r_id = gr.Number(label="review_id", precision=0)
                r_dec = gr.Radio(["confirm", "adjust", "reject"], value="confirm", label="Решение")
                r_adj = gr.Slider(0, 1, value=0.5, label="Скорректированный риск")
            r_comment = gr.Textbox(label="Комментарий преподавателя (например, личные обстоятельства)")
            r_out = gr.Markdown()
            gr.Button("Сохранить", variant="primary").click(submit_review, [token, r_id, r_dec, r_adj, r_comment], r_out)
        with gr.Tab("Оценка качества"):
            with gr.Row():
                c_models = gr.CheckboxGroup(MODELS_CLF, value=MODELS_CLF, label="Модели (UCI, G3<10)")
                c_size = gr.Number(value=0, label="Размер выборки (0 — все)")
            c_table, c_plot, c_tests = gr.Dataframe(), gr.Plot(), gr.Dataframe(label="Тесты значимости")
            gr.Button("Сравнить").click(run_compare, [token, c_models, c_size], [c_table, c_plot, c_tests])
            gr.Markdown("### Пакетная оценка режимов")
            with gr.Row():
                b_n = gr.Number(value=3, label="Число наборов")
                b_modes = gr.CheckboxGroup(["baseline", "+rag", "+agents"], value=["baseline", "+rag"], label="Режимы")
                b_fr = gr.Textbox(value="0.2,0.5,1.0", label="Доли данных")
            b_job = gr.Textbox(label="job_id")
            gr.Button("Запустить оценку").click(start_batch, [token, b_n, b_modes, b_fr], b_job)
            b_status, b_table, b_plot = gr.Markdown(), gr.Dataframe(), gr.Plot()
            gr.Button("Обновить").click(batch_status, [token, b_job], [b_status, b_table, b_plot])
        with gr.Tab("Логи агентов"):
            l_task = gr.Textbox(label="task_id")
            l_auto = gr.Checkbox(value=False, label="Автообновление (реальное время)")
            l_md, l_plot = gr.Markdown(), gr.Plot()
            l_calls = gr.Dataframe(label="Вызовы LLM")
            l_metrics = gr.Code(language="json", label="Агентные метрики")
            l_outs = [l_md, l_plot, l_calls, l_metrics]
            gr.Button("Показать").click(load_logs, [token, l_task], l_outs)
            gr.Timer(2.0).tick(lambda t, tid, a: load_logs(t, tid) if a and tid else (gr.skip(),) * 4,
                               [token, l_task, l_auto], l_outs)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name=os.environ.get("MLF_UI_HOST", "127.0.0.1"),
                      server_port=int(os.environ.get("MLF_UI_PORT", "7860")))
