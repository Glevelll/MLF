"""Gradio workspace for forecasting, review, evaluation and agent logs."""
from __future__ import annotations

import os
from datetime import datetime
from html import escape
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px

from frontend.api_client import ApiClient, ApiError
from frontend.presentation import (
    METRICS,
    MODE_LABELS,
    MODELS,
    STATUSES,
    banner,
    batch_plot,
    chart,
    fmt,
    importance_plot,
    metrics_table,
    predictions_table,
    risk_plot,
    summary_cards,
    timeline,
    tree_markdown,
)

DATASETS = [("UCI · Математика", "uci_mat"), ("UCI · Португальский язык", "uci_por"),
            ("UCI · Оба предмета", "uci_both"), ("Синтетические данные", "synthetic")]
SOURCE_NAMES = {**{key: name for name, key in DATASETS}, "input.csv": "Загруженная таблица CSV",
                "input.parquet": "Загруженная таблица Parquet"}
MODELS_CLF = list(MODELS)[:3]
MODELS_REG = ["fit_ridge_regression", "fit_random_forest", "fit_tabular_xgboost"]
ROLE_NAMES = {"user": "Пользователь", "editor": "Редактор", "admin": "Администратор"}
RUNNING = {"pending", "running"}
TASK_ACTIONS = [("Принять результат", "accept"), ("Выбрать другую модель", "choose_model"),
                ("Сменить порог вероятности", "change_threshold"), ("Отклонить", "reject")]
DATA_PAGE_SIZE = 25
COLUMN_NAMES = {
    "student_id": "код студента", "attendance_rate": "доля посещённых занятий, 0,8 означает 80 %",
    "hw_submitted_ratio": "доля сданных домашних заданий", "hw_avg_score": "средний балл за домашние задания",
    "test1_score": "балл за первый тест", "test2_score": "балл за второй тест",
    "lms_hours": "часы работы на учебной платформе", "late_submission_rate": "доля просроченных заданий",
    "study_format": "формат обучения, онлайн или смешанный", "final_score": "фактический итоговый балл",
    "G1": "балл за первый учебный период", "G2": "балл за второй учебный период", "G3": "итоговый балл",
}
THEME_TOGGLE = """() => {
    const dark = document.body.classList.toggle('dark');
    localStorage.setItem('mlf-theme', dark ? 'dark' : 'light');
}"""
THEME_INIT = """() => {
    const saved = localStorage.getItem('mlf-theme');
    if (saved) document.body.classList.toggle('dark', saved === 'dark');
}"""


def client(token):
    if not token:
        raise ApiError("Войдите, чтобы продолжить.", 401)
    return ApiClient(token=token)


def err(error):
    return banner("Не удалось выполнить действие", error, "error")


def sign_in(username, password):
    if not (username or "").strip() or not password:
        return *(gr.skip(),) * 6, err("Введите логин и пароль.")
    try:
        api = ApiClient()
        access = api.login(username.strip(), password)["access_token"]
        user = api.me()
        name, role = user["username"], user["role"]
        profile = (f'<div class="profile"><span class="avatar">{escape(name[:1].upper())}</span>'
                   f'<div><strong>{escape(name)}</strong><small>{escape(ROLE_NAMES.get(role, role))}</small></div></div>')
        return (access, gr.update(visible=False), gr.update(visible=True), profile,
                gr.update(visible=role in {"editor", "admin"}), gr.update(visible=True), "")
    except ApiError as exc:
        return *(gr.skip(),) * 6, err(exc)


def register(email, username, password):
    name = (username or "").strip()
    if not 3 <= len(name) <= 50:
        return err("Имя пользователя должно содержать от 3 до 50 символов.")
    if not 6 <= len(password or "") <= 72:
        return err("Пароль должен содержать от 6 до 72 символов.")
    if "@" not in (email or ""):
        return err("Проверьте адрес электронной почты.")
    try:
        ApiClient().register(email.strip(), name, password)
        return banner("Аккаунт создан", "Теперь войдите с тем же логином и паролем.", "success")
    except ApiError as exc:
        return err(exc)


def forecast_form(source, dataset, problem, use_cutoff):
    upload = source == "file"
    numeric_target = (upload and use_cutoff) or (not upload and dataset != "synthetic")
    hints = {
        "regression": "Модель рассчитает итоговый балл для каждого студента.",
        "classification": "Модель рассчитает вероятность попадания в группу риска.",
        "auto": "Для файла тип прогноза определяется по целевой колонке.",
    }
    return (gr.update(visible=not upload), gr.update(visible=upload), gr.update(visible=upload),
            gr.update(visible=upload and problem != "regression"),
            gr.update(visible=numeric_target and problem != "regression"), hints[problem])


def forecast_algorithms(problem):
    models = {"classification": MODELS_CLF, "regression": MODELS_REG}.get(problem, [])
    choices = [("Агенты выберут", "auto"), *((MODELS[key], key) for key in models)]
    return gr.update(choices=choices, value="auto", interactive=bool(models))


def ollama_options(token, provider):
    if provider != "ollama":
        return gr.update(visible=False, choices=[], value=None), gr.update(visible=False), ""
    try:
        data = client(token).agent_models()
        message = "" if data["models"] else err("В Ollama пока нет скачанной Qwen.")
        return gr.update(visible=True, choices=data["models"], value=data["preferred"]), gr.update(visible=True), message
    except ApiError as exc:
        return gr.update(visible=True, choices=[], value=None), gr.update(visible=True), err(exc)


def forecast_payload(source, file, dataset, problem, target, use_cutoff, cutoff, iterations):
    if source == "file" and (not file or not (target or "").strip()):
        raise ValueError("Загрузите файл и укажите название целевой колонки.")
    params = {"problem_type": problem, "max_iterations": int(iterations)}
    if source == "file":
        params["target_column"] = target.strip()
        apply_cutoff = use_cutoff and problem != "regression"
    else:
        params["dataset"] = dataset
        if dataset == "synthetic":
            params["target_column"] = "final_score" if problem == "regression" else "dropout"
        else:
            params["target_column"] = "G3"
        apply_cutoff = dataset != "synthetic" and problem != "regression"
        if problem == "auto":
            params["problem_type"] = "classification"
    if apply_cutoff:
        if cutoff is None:
            raise ValueError("Укажите балл, ниже которого студент попадает в группу риска.")
        params["binarize_threshold"] = cutoff
    return params


def status_update(title, detail="", kind="info"):
    return (banner(title, detail, kind), gr.update(visible=False), gr.update(visible=False), "", None, None, None,
            None, gr.update(visible=False), "")


def source_name(source):
    return SOURCE_NAMES.get(source, source or "Исходная таблица")


def history_choices(token, selected=None):
    if not token:
        return gr.skip(), gr.skip()
    try:
        items = client(token).tasks()
        choices = []
        for item in items:
            date = datetime.fromisoformat(item["created_at"]).strftime("%d.%m %H:%M")
            origin = source_name(item.get("source"))
            target = f" → {item['target_column']}" if item.get("target_column") else ""
            status = STATUSES.get(item["status"], item["status"])
            choices.append((f"{date} · {origin}{target} · {status} · {item['task_id'][:8]}", item["task_id"]))
        picked = selected if any(value == selected for _, value in choices) else None
        return gr.update(choices=choices, value=picked), "" if choices else "История пока пуста."
    except ApiError as exc:
        return gr.skip(), str(exc)


def start_forecast(token, source, file, dataset, problem, target, use_cutoff, cutoff, iterations,
                   agent_provider, agent_model, forecast_model):
    try:
        params = forecast_payload(source, file, dataset, problem, target, use_cutoff, cutoff, iterations)
        params["agent_provider"] = agent_provider
        if agent_provider == "ollama":
            if not agent_model:
                raise ValueError("Выберите модель Qwen. Если список пуст, обновите его.")
            params["agent_model"] = agent_model
        if forecast_model != "auto":
            params["requested_model"] = forecast_model
        res = client(token).create_forecast(file if source == "file" else None, **params)
        return res["task_id"], gr.update(active=True), "", *status_update(
            "Прогноз поставлен в очередь", "Здесь появятся модель, метрики и таблица студентов."
        )
    except (ApiError, ValueError) as exc:
        return gr.skip(), gr.skip(), err(exc), *(gr.skip(),) * 10


def result_notes(result):
    parts = []
    if result.get("critic_reasoning"):
        parts.append(f"**Комментарий критика**\n\n{escape(result['critic_reasoning'])}")
    issues = result.get("issues") or []
    if issues:
        parts.append("**Замечания валидатора**\n\n" + "\n".join(
            f"- {escape(item.get('message') or item.get('code', ''))}" for item in issues))
    actions = result.get("editor_actions") or []
    if actions:
        parts.append("**Правки редактора**\n\n" + "\n".join(f"- {escape(action['tool'])}" for action in actions))
    cases = result.get("rag_cases") or []
    if cases:
        parts.append(f"**Поиск по базе знаний**\n\nПодобрано похожих кейсов: {len(cases)}.")
    return "\n\n".join(parts) or "Замечаний по этому прогнозу нет."


def refresh_forecast(token, task_id):
    if not token or not task_id:
        return (gr.skip(),) * 11
    try:
        api = client(token)
        status = api.status(task_id)
        state = status["status"]
        name = STATUSES.get(state, state)
        if state in RUNNING:
            detail = f"Итерация {status['iteration']}" if status.get("iteration") else "Подготавливаем данные."
            return gr.update(active=True), *status_update(name, detail)
        if state == "failed":
            return gr.update(active=False), *status_update(
                name, status.get("error") or "Проверьте данные и попробуйте снова.", "error")
        result = api.result(task_id, preview=30)
        pending = state == "waiting_for_human"
        detail = (status.get("hitl_reason") or "Доступен предварительный результат.") if pending else (
            "Откройте таблицу студентов или посмотрите, как работает модель.")
        export = gr.update(visible=False, value=None)
        if not pending:
            try:
                export = gr.update(visible=True, value=api.download_predictions(task_id))
            except ApiError:
                detail = "CSV пока недоступен. Попробуйте обновить статус."
        context = []
        if status.get("source"):
            context.append(f"{source_name(status['source'])} · {status.get('target_column') or 'прогноз'}.")
        if status.get("agent_provider"):
            context.append(f"Агенты — {status.get('agent_model') or status['agent_provider']}.")
        detail = " ".join([*context, detail])
        return (gr.update(active=False), banner(name, detail, "warning" if pending else "success"),
                gr.update(visible=True), gr.update(visible=pending), summary_cards(result), metrics_table(result),
                importance_plot(result.get("feature_importance")), risk_plot(result.get("risk_counts")),
                predictions_table(result), export, result_notes(result))
    except ApiError as exc:
        if exc.status == 409:
            return gr.update(active=True), *status_update("Идёт расчёт")
        return gr.update(active=False), *status_update("Не удалось получить прогноз", str(exc), "error")


def load_queue(token, task_id):
    try:
        rows = client(token).queue(kind="student_risk", task_id=(task_id or "").strip() or None)
        entries, choices = [], []
        for item in rows:
            payload = item["payload"]
            student = str(payload.get("student_id", ""))
            entries.append({"Студент": student, "Оценка модели": fmt(payload.get("score"), 2),
                            "Факторы": ", ".join(d["feature"] for d in payload.get("drivers", [])),
                            "Рекомендации": " · ".join(payload.get("recommendations", [])[:2])})
            choices.append((f"{student} · {item['task_id'][:8]} · запись {item['id']}", str(item["id"])))
        return pd.DataFrame(entries), gr.update(choices=choices, value=None), banner(
            f"На проверке: {len(choices)}", "Выберите студента из списка." if choices else "Новых записей нет.")
    except ApiError as exc:
        return pd.DataFrame(), gr.update(choices=[], value=None), err(exc)


def submit_review(token, review_id, decision, adjusted, comment):
    if not review_id:
        return err("Выберите студента из очереди.")
    try:
        res = client(token).review(int(review_id), decision, adjusted if decision == "adjust" else None, comment)
        detail = "Случай добавлен в базу знаний." if res.get("kb_added") else ""
        return banner("Решение сохранено", detail, "success")
    except ApiError as exc:
        return err(exc)


def submit_task_hitl(token, task_id, action, model, threshold, comment):
    if not (task_id or "").strip():
        return err("Откройте прогноз, который нужно проверить."), gr.skip(), gr.skip()
    if action == "choose_model" and not model:
        return err("Выберите модель."), gr.skip(), gr.skip()
    try:
        task_id = task_id.strip()
        client(token).hitl(task_id, action=action, model_type=model if action == "choose_model" else None,
                           threshold=threshold if action == "change_threshold" else None, comment=comment)
        message = banner("Решение принято", "Статус обновится в разделе «Прогнозы».", "success")
        return message, task_id, gr.update(active=True)
    except ApiError as exc:
        return err(exc), gr.skip(), gr.skip()


def comparison_result(res):
    metric = METRICS.get(res["metric"], res["metric"])
    scores = pd.DataFrame([{"Модель": MODELS.get(m, m), metric: r["fold_mean"], "Разброс": r["fold_std"]}
                           for m, r in res["results"].items()]).sort_values(metric, ascending=False)
    fig = px.bar(scores, x="Модель", y=metric, error_y="Разброс")
    fig.update_traces(marker_color="#087f78")
    tests = pd.DataFrame([{"Пара": f"{MODELS.get(t['a'], t['a'])} / {MODELS.get(t['b'], t['b'])}",
                           "Прирост": t["mean_improvement"], "Уилкоксон": t["wilcoxon"]["pvalue"],
                           "Парный t-тест": t["ttest"]["pvalue"], "Диболд–Мариано": t["diebold_mariano"]["pvalue"]}
                          for t in res.get("tests", [])])
    return banner("Сравнение готово", f"Лучшая по {metric}: {MODELS.get(res['winner'], res['winner'])}.",
                  "success"), chart(fig, height=360), scores, tests


def compare(token, models, sample_size):
    try:
        if len(models) < 2:
            raise ValueError("Выберите минимум две модели.")
        if sample_size and sample_size < 50:
            raise ValueError("Укажите не менее 50 записей или 0 для всей таблицы.")
        res = client(token).compare(models, int(sample_size) if sample_size else None)
        return res["comparison_id"], *comparison_result(res)
    except (ApiError, ValueError) as exc:
        return gr.skip(), err(exc), gr.skip(), gr.skip(), gr.skip()


def history_dropdown(choices, selected):
    ids = {value for _, value in choices}
    value = selected if selected in ids else choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def quality_history(token, comparison_id, batch_id):
    if not token:
        return (gr.skip(),) * 3
    try:
        data = client(token).evaluation_history()
        comparisons = []
        for row in data["comparisons"]:
            created = datetime.fromisoformat(row["created_at"]).strftime("%d.%m %H:%M")
            winner = MODELS.get(row["winner"], row["winner"] or "—")
            comparisons.append((f"{created} · {row['sample_size']} строк · {winner} · №{row['id']}", row["id"]))
        batches = []
        for row in data["batches"]:
            created = datetime.fromisoformat(row["created_at"]).strftime("%d.%m %H:%M")
            count = (row.get("params") or {}).get("num_datasets", "—")
            batches.append((f"{created} · Наборов: {count} · {row['id'][:8]}", row["id"]))
        return history_dropdown(comparisons, comparison_id), history_dropdown(batches, batch_id), ""
    except ApiError as exc:
        return gr.skip(), gr.skip(), err(exc)


def open_comparison(token, comparison_id):
    if not token or comparison_id is None:
        return (gr.skip(),) * 5
    try:
        return comparison_id, *comparison_result(client(token).saved_comparison(comparison_id))
    except ApiError as exc:
        return gr.skip(), err(exc), gr.skip(), gr.skip(), gr.skip()


def start_batch(token, count, modes, shares):
    if not count or not modes or not shares:
        return gr.skip(), gr.skip(), err("Выберите число наборов, режимы и доли обучения."), gr.skip(), gr.skip()
    try:
        job = client(token).batch(num_datasets=int(count), modes=modes, fractions=[float(value) for value in shares])
        return job["task_id"], gr.update(active=True), banner("Оценка поставлена в очередь"), None, None
    except (ApiError, ValueError) as exc:
        return gr.skip(), gr.skip(), err(exc), gr.skip(), gr.skip()


def refresh_batch(token, job):
    if not token or not job:
        return (gr.skip(),) * 4
    try:
        res = client(token).batch_status(job)
        state = res["status"]
        if state in RUNNING:
            return gr.update(active=True), banner("Идёт оценка", "Большая выборка может занять несколько минут."), None, None
        if state == "failed":
            return gr.update(active=False), err(res.get("error") or "Пакетная оценка завершилась с ошибкой."), None, None
        summary = res.get("summary") or {}
        groups = pd.DataFrame(summary.get("groups") or [])
        fig = batch_plot(groups)
        if not groups.empty:
            groups["mode"] = groups["mode"].replace(MODE_LABELS)
            groups["problem_type"] = groups["problem_type"].replace({"classification": "Риск", "regression": "Оценка"})
            groups["fraction"] = groups["fraction"].map(lambda value: f"{value:.0%}")
        cols = {"problem_type": "Задача", "mode": "Режим", "fraction": "Доля обучения", "m_roc_auc": "ROC-AUC",
                "m_r2": "R²", "m_f1": "F1", "m_mae": "MAE", "time_s": "Время, с"}
        failures = summary.get("errors", 0)
        detail = f"Успешных запусков: {summary.get('successful_runs', 0)}. С ошибками: {failures}."
        table = groups[[key for key in cols if key in groups]].rename(columns=cols).round(4)
        return gr.update(active=False), banner("Оценка готова", detail, "warning" if failures else "success"), fig, table
    except ApiError as exc:
        return gr.update(active=False), err(exc), None, None


def open_batch(token, job):
    if not token or not job:
        return (gr.skip(),) * 5
    return job, *refresh_batch(token, job)


def load_logs(token, task_id):
    if not (task_id or "").strip():
        return banner("Выберите прогноз", "Откройте журнал из результата или введите идентификатор."), "", None, None
    try:
        res = client(token).reasoning(task_id.strip())
        head = banner(f"Шагов: {res['n_spans']}", f"Токенов в запросах к языковой модели: {res['total_tokens']}.")
        cols = {"agent": "Агент", "provider": "Провайдер", "model": "Модель", "tokens": "Токены",
                "latency_ms": "Время, мс", "status": "Статус", "cached": "Из кэша"}
        calls = pd.DataFrame(res["llm_calls"])
        calls = calls[[key for key in cols if key in calls]].rename(columns=cols)
        return head, "\n\n".join(tree_markdown(res["tree"])), timeline(res["tree"]), calls
    except ApiError as exc:
        return err(exc), "", None, None


def agent_overview(token):
    try:
        stats = client(token).agent_metrics()
        rate = stats["hitl_rate"]
        rows = {
            "Дошли до результата или ручной проверки": stats["tasks_finished"],
            "Среднее число итераций": fmt(stats["avg_iterations"], 2),
            "Доля задач на ручной проверке": "—" if rate is None else f"{fmt(rate * 100, 1)} %",
            "Всего вызовов языковой модели": stats["llm_calls"],
            "Среднее время вызова, мс": fmt(stats["avg_llm_latency_ms"], 0),
        }
        return pd.DataFrame(rows.items(), columns=["Показатель", "Значение"]), ""
    except ApiError as exc:
        return None, err(exc)


def section(title, text=""):
    gr.Markdown(f"## {title}" + (f"\n\n{text}" if text else ""), elem_classes="section-heading")


def table(label):
    return gr.Dataframe(label=label, interactive=False, wrap=True, max_height=380, elem_classes="data-table")


def row_range(offset, count, total):
    if not count:
        return "На этой странице нет строк." if total else "Здесь пока нет записей."
    return f"Строки {offset + 1}–{offset + count} из {total}."


def load_input_preview(token, task_id, subset, page):
    if not task_id:
        return banner("Выберите прогноз", "В списке есть завершённые и текущие запуски."), None, ""
    offset = (max(1, int(page or 1)) - 1) * DATA_PAGE_SIZE
    try:
        data = client(token).input_data(task_id, subset, offset, DATA_PAGE_SIZE)
        title = source_name(data["source"])
        text = (f"Целевая колонка — {data['target_column']}. С известным значением — {data['known_rows']}, "
                f"для прогноза — {data['new_rows']}. {row_range(offset, len(data['rows']), data['matching_rows'])}")
        details = []
        if data["model"] and data["features"]:
            features = ", ".join(escape(name) for name in data["features"])
            details.append(f"**Колонки, которые использовала модель.** {features}.")
        details.extend(f"**{name}** — {label}." for name, label in COLUMN_NAMES.items() if name in data["columns"])
        return banner(title, text), pd.DataFrame(data["rows"], columns=data["columns"]), "\n\n".join(details)
    except ApiError as exc:
        return err(exc), None, ""


def open_active_data(token, task_id):
    choices, note = history_choices(token, task_id)
    return gr.update(selected="data"), gr.update(selected="source"), choices, note, "all", 1


def load_knowledge(token, kind, page):
    offset = (max(1, int(page or 1)) - 1) * DATA_PAGE_SIZE
    try:
        api = client(token)
        stats = api.kb_stats()
        count = sum(stats["knowledge_base"].values())
        summary = banner("База знаний", f"Кейсов — {count}. Терминов — {stats['glossary']}.")
    except ApiError as exc:
        return err(exc), "", None
    try:
        records = api.kb_records(kind, offset, DATA_PAGE_SIZE)
        columns = {"source_id": "Код кейса", "description": "Описание", "target_column": "Цель",
                   "best_model": "Лучшая модель", "term": "Термин", "definition": "Определение", "category": "Раздел"}
        frame = pd.DataFrame(records["rows"])
        if "best_model" in frame:
            frame["best_model"] = frame["best_model"].replace(MODELS)
        return summary, row_range(offset, len(frame), records["total"]), frame.rename(columns=columns)
    except ApiError as exc:
        message = "Для просмотра записей нужна роль редактора или администратора." if exc.status == 403 else str(exc)
        return summary, message, None


def build_ui():
    with gr.Blocks(title="MLF · Прогноз успеваемости", analytics_enabled=False, fill_width=True,
                   elem_id="mlf-app") as demo:
        token, active_id, job_id = gr.State(None), gr.State(""), gr.State("")
        comparison_id = gr.State(None)
        forecast_timer = gr.Timer(3, active=False, render=False)
        batch_timer = gr.Timer(4, active=False, render=False)
        log_timer = gr.Timer(5, active=False, render=False)

        with gr.Row(elem_classes="masthead"):
            gr.HTML('<div class="brand"><span class="brand-mark">M</span><span><strong>MLF</strong>'
                    '<small>Прогноз успеваемости</small></span></div>')
            with gr.Row(elem_classes="account-bar"):
                profile = gr.HTML()
                theme_btn = gr.Button("Тема", size="sm", elem_classes="small-button")
                logout = gr.Button("Выйти", size="sm", visible=False, elem_classes="small-button")
        with gr.Row(elem_classes="auth-grid") as auth:
            gr.HTML(
                '<div class="login-intro"><div class="intro-chart" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>'
                '<h1>Прогноз успеваемости студентов</h1><p>Оцените риск неуспеваемости или будущую оценку. '
                'Посмотрите, какие данные повлияли на прогноз.</p></div>',
                elem_classes="login-intro-shell",
            )
            with gr.Column(elem_classes="card auth-card"):
                with gr.Tabs(elem_classes="subtabs"):
                    with gr.Tab("Вход"):
                        section("Вход в MLF")
                        username = gr.Textbox(label="Логин или email", autofocus=True)
                        password = gr.Textbox(label="Пароль", type="password")
                        login_btn = gr.Button("Войти", variant="primary")
                        login_msg = gr.HTML()
                    with gr.Tab("Регистрация"):
                        section("Создать аккаунт")
                        email = gr.Textbox(label="Email")
                        name = gr.Textbox(label="Имя пользователя", placeholder="От 3 символов")
                        secret = gr.Textbox(label="Пароль", type="password", placeholder="От 6 символов")
                        reg = gr.Button("Зарегистрироваться", variant="primary")
                        reg_msg = gr.HTML()

        with gr.Column(visible=False, elem_classes="workspace") as workspace:
            with gr.Tabs(selected="forecast", overflow_behavior="wrap", elem_id="workspace-tabs") as navigation:
                with gr.Tab("Прогнозы", id="forecast", elem_id="nav-forecast"):
                    with gr.Row(elem_classes="page-heading"):
                        section("Прогнозы", "Выберите данные и задачу. Результат появится рядом с настройками.")
                    with gr.Row(elem_classes="dashboard-grid"):
                        with gr.Column(scale=1, min_width=290, elem_classes="card forecast-form"):
                            section("Новый прогноз")
                            source = gr.Radio([("Готовые данные", "builtin"), ("Свой файл", "file")], value="builtin",
                                              label="Источник данных", elem_classes="pill-field")
                            dataset = gr.Dropdown(DATASETS, value="uci_mat", label="Набор данных")
                            file = gr.File(label="Таблица CSV или Parquet", type="filepath", visible=False,
                                           file_types=[".csv", ".parquet", ".pq"])
                            target = gr.Textbox(label="Целевая колонка", placeholder="Например, G3", visible=False)
                            problem = gr.Dropdown([("Риск неуспеваемости", "classification"),
                                                   ("Итоговая оценка", "regression"), ("Определить автоматически", "auto")],
                                                  value="classification", label="Что прогнозируем")
                            hint = gr.Markdown("Модель рассчитает вероятность попадания в группу риска.",
                                               elem_classes="field-hint")
                            use_cutoff = gr.Checkbox(label="Получить классы из числовой оценки", visible=False)
                            cutoff = gr.Number(value=10, label="Балл, ниже которого студент в группе риска")
                            agent_provider = gr.Dropdown([("Mock · быстрые правила", "mock"), ("Ollama · Qwen", "ollama")],
                                                         value="mock", label="Кто помогает агентам")
                            with gr.Row(elem_classes="search-row"):
                                agent_model = gr.Dropdown([], label="Модель Qwen", interactive=True, visible=False)
                                reload_models = gr.Button("Обновить модели", size="sm", visible=False)
                            agent_message = gr.HTML()
                            forecast_model = gr.Dropdown([("Агенты выберут", "auto"),
                                                          *((MODELS[key], key) for key in MODELS_CLF)],
                                                         value="auto", label="Алгоритм прогноза")
                            launch = gr.Button("Рассчитать прогноз", variant="primary")
                            form_msg = gr.HTML()
                            with gr.Accordion("Дополнительные настройки", open=False):
                                iterations = gr.Slider(1, 5, step=1, value=3, label="Максимум итераций агентов")
                            with gr.Accordion("Недавние прогнозы", open=False) as history_box:
                                history = gr.Dropdown([], label="История", show_label=False, interactive=True,
                                                      container=False)
                                history_note = gr.Markdown()
                                reload_history = gr.Button("Обновить список", size="sm")
                        with gr.Column(scale=2, min_width=300, elem_classes="card result-card"):
                            with gr.Row(elem_classes="result-heading"):
                                section("Результат")
                                refresh = gr.Button("Обновить статус", size="sm", elem_classes="small-button")
                            status = gr.HTML(banner("Здесь появится прогноз",
                                                    "Выберите готовый набор или загрузите таблицу, затем запустите расчёт."),
                                             elem_classes="result-notice")
                            with gr.Column(visible=False) as result_content:
                                cards = gr.HTML()
                                with gr.Row(elem_classes="compact-actions"):
                                    open_source = gr.Button("На чём обучалась модель", size="sm")
                                    open_log = gr.Button("Посмотреть шаги агентов", size="sm")
                                    open_review = gr.Button("Открыть ручную проверку", visible=False, size="sm")
                                    export = gr.DownloadButton("Скачать CSV", size="sm", visible=False)
                                with gr.Tabs(elem_classes="subtabs"):
                                    with gr.Tab("Студенты"):
                                        gr.Markdown("До 30 записей. Полная таблица доступна в CSV после завершения задачи.")
                                        predictions = table("Предпросмотр прогноза")
                                    with gr.Tab("Качество и факторы"):
                                        metrics = table("Метрики качества")
                                        with gr.Row(elem_classes="chart-grid"):
                                            with gr.Column(min_width=260):
                                                gr.Markdown("**Важность признаков**")
                                                importance = gr.Plot(show_label=False)
                                            with gr.Column(min_width=260):
                                                gr.Markdown("**Студенты по уровню риска**")
                                                risks = gr.Plot(show_label=False)
                                    with gr.Tab("Замечания агентов"):
                                        notes = gr.Markdown()

                with gr.Tab("Данные", id="data", elem_id="nav-data") as data_tab:
                    with gr.Row(elem_classes="page-heading"):
                        section("Данные", "Откройте таблицу прогноза или примеры, которые ищут агенты в базе знаний.")
                    with gr.Tabs(selected="source", elem_classes="subtabs") as data_tabs:
                        with gr.Tab("Исходная таблица", id="source"):
                            with gr.Column(elem_classes="card"):
                                data_task = gr.Dropdown([], label="Какой прогноз смотрим", interactive=True)
                                data_history_note = gr.Markdown()
                                with gr.Row(elem_classes="search-row"):
                                    data_subset = gr.Dropdown([("Все строки", "all"), ("Известен ответ", "known"),
                                                               ("Для нового прогноза", "new")], value="all", label="Строки")
                                    data_page = gr.Number(1, label="Страница", precision=0, minimum=1)
                                    data_load = gr.Button("Показать таблицу", variant="primary")
                                data_head = gr.HTML(banner("Выберите прогноз"))
                                gr.Markdown("Строки с заполненной целью нужны для обучения и проверки. "
                                            "Если цель пустая, модель рассчитывает прогноз для этой строки.")
                                source_rows = table("Исходные значения")
                                with gr.Accordion("Что означают колонки", open=False):
                                    data_notes = gr.Markdown()
                        with gr.Tab("База знаний", id="kb") as kb_tab:
                            with gr.Column(elem_classes="card"):
                                gr.Markdown("Кейсы помогают агентам подобрать алгоритм. Здесь можно посмотреть записи "
                                            "базы знаний и справочник терминов.")
                                kb_summary = gr.HTML()
                                with gr.Row(elem_classes="search-row"):
                                    kb_kind = gr.Dropdown([("Подбор модели", "dataset_case"),
                                                           ("Примеры студентов", "student_case"), ("Справочник", "glossary")],
                                                          value="dataset_case", label="Что показать")
                                    kb_page = gr.Number(1, label="Страница", precision=0, minimum=1)
                                    kb_load = gr.Button("Показать записи", variant="primary")
                                kb_note = gr.Markdown()
                                kb_table = table("Записи базы знаний")

                with gr.Tab("Ручная проверка", id="review", elem_id="nav-review"):
                    with gr.Row(elem_classes="page-heading"):
                        section("Ручная проверка", "Подтвердите предварительный прогноз или разберите записи группы риска.")
                    with gr.Row(elem_classes="dashboard-grid review-grid"):
                        with gr.Column(scale=1, min_width=295, elem_classes="card"):
                            section("Решение по прогнозу", "Доступно, когда агенты передали задачу на ручную проверку.")
                            checked_task = gr.Textbox(label="Идентификатор прогноза")
                            use_active = gr.Button("Использовать текущий прогноз", size="sm")
                            action = gr.Dropdown(TASK_ACTIONS, value="accept", label="Что сделать")
                            revised_model = gr.Dropdown([(v, k) for k, v in MODELS.items()], label="Модель", visible=False)
                            risk_threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Порог вероятности",
                                                       visible=False)
                            task_comment = gr.Textbox(label="Комментарий (необязательно)")
                            decide = gr.Button("Сохранить решение", variant="primary")
                            task_feedback = gr.HTML()
                        with gr.Column(scale=2, min_width=300, visible=False, elem_classes="card") as editor_panel:
                            section("Очередь студентов", "Здесь собраны записи, которые модель отнесла к высокому риску.")
                            queue_filter = gr.Textbox(label="Фильтр по прогнозу (необязательно)")
                            with gr.Row(elem_classes="compact-actions"):
                                queue_reload = gr.Button("Загрузить очередь", size="sm")
                                queue_active = gr.Button("Только текущий прогноз", size="sm")
                            queue_note = gr.HTML()
                            queue_table = table("Студенты на проверке")
                            student = gr.Dropdown([], label="Студент для проверки", interactive=True)
                            decision = gr.Radio([("Подтвердить", "confirm"), ("Уточнить", "adjust"), ("Отклонить", "reject")],
                                                value="confirm", label="Решение", elem_classes="pill-field")
                            adjusted = gr.Slider(0, 1, step=0.01, value=0.5, label="Уточнённая вероятность", visible=False)
                            student_comment = gr.Textbox(label="Комментарий (необязательно)")
                            save_review = gr.Button("Сохранить проверку", variant="primary")
                            review_feedback = gr.HTML()

                with gr.Tab("Оценка качества", id="quality", elem_id="nav-quality") as quality_tab:
                    with gr.Row(elem_classes="page-heading"):
                        section("Оценка качества", "Сравните алгоритмы на UCI или режимы работы на нескольких наборах.")
                        refresh_quality = gr.Button("Обновить историю", size="sm", scale=0)
                    quality_history_note = gr.HTML()
                    with gr.Tabs(elem_classes="subtabs compare-tabs"):
                        with gr.Tab("Сравнение моделей"):
                            with gr.Row(elem_classes="dashboard-grid"):
                                with gr.Column(scale=1, min_width=290, elem_classes="card"):
                                    section("Настройки сравнения", "Данные UCI, целевая оценка G3 ниже 10.")
                                    compare_models = gr.CheckboxGroup([(MODELS[key], key) for key in MODELS_CLF],
                                                                     value=MODELS_CLF, label="Модели")
                                    with gr.Accordion("Размер выборки", open=False):
                                        sample = gr.Number(label="Число записей (0 — все)", value=0, precision=0)
                                    compare_btn = gr.Button("Сравнить модели", variant="primary")
                                with gr.Column(scale=2, min_width=300, elem_classes="card"):
                                    section("Результаты сравнения", "Чем ближе ROC-AUC к 1, тем лучше модель расставляет риск.")
                                    comparison_history = gr.Dropdown(choices=[], label="Сохранённые сравнения",
                                                                     interactive=True)
                                    compare_msg = gr.HTML(banner("Сравнение ещё не запущено"))
                                    compare_fig = gr.Plot(show_label=False)
                                    compare_scores = table("Метрики")
                                    with gr.Accordion("Статистические тесты", open=False):
                                        compare_tests = table("p-значения тестов")
                        with gr.Tab("Пакетная оценка"):
                            with gr.Row(elem_classes="dashboard-grid"):
                                with gr.Column(scale=1, min_width=290, elem_classes="card"):
                                    section("Настройки эксперимента")
                                    count = gr.Dropdown([("3 · быстрая проверка", 3), ("20 наборов", 20), ("50 наборов", 50),
                                                         ("100 наборов", 100)], value=3, label="Размер эксперимента")
                                    modes = gr.CheckboxGroup([(v, k) for k, v in MODE_LABELS.items()],
                                                             value=["baseline", "+rag"], label="Режимы")
                                    shares = gr.CheckboxGroup([("10 %", "0.1"), ("20 %", "0.2"), ("50 %", "0.5"),
                                                              ("100 %", "1.0")], value=["1.0"], label="Доли обучения")
                                    batch_btn = gr.Button("Запустить оценку", variant="primary")
                                with gr.Column(scale=2, min_width=300, elem_classes="card"):
                                    section("Результаты эксперимента")
                                    batch_history = gr.Dropdown(choices=[], label="Сохранённые оценки",
                                                                interactive=True)
                                    batch_msg = gr.HTML(banner("Оценка ещё не запущена"))
                                    batch_fig = gr.Plot(show_label=False)
                                    batch_scores = table("Метрики по наборам")

                with gr.Tab("Журнал агентов", id="logs", elem_id="nav-logs"):
                    with gr.Row(elem_classes="page-heading"):
                        section("Журнал агентов", "Посмотрите, какую модель выбрали агенты и какие проверки выполнили.")
                    with gr.Column(elem_classes="card"):
                        with gr.Row(elem_classes="search-row"):
                            log_id = gr.Textbox(label="Идентификатор прогноза", placeholder="Открывается из результата")
                            logs_btn = gr.Button("Показать журнал", variant="primary")
                            log_active = gr.Button("Текущий прогноз")
                        log_live = gr.Checkbox(label="Обновлять журнал каждые 5 секунд")
                        log_head = gr.HTML(banner("Журнал появится здесь", "Откройте его из результата прогноза."))
                        with gr.Tabs(elem_classes="subtabs"):
                            with gr.Tab("Действия"):
                                log_lines = gr.Markdown()
                            with gr.Tab("Время по шагам"):
                                log_timeline = gr.Plot(show_label=False)
                            with gr.Tab("Вызовы модели"):
                                log_calls = table("Вызовы языковой модели")
                            with gr.Tab("Сводка"):
                                overview_btn = gr.Button("Загрузить сводку", size="sm")
                                overview_status = gr.HTML()
                                overview = table("Метрики по всем прогнозам")
            forecast_timer.render()
            batch_timer.render()
            log_timer.render()

        access_outputs = [token, auth, workspace, profile, editor_panel, logout, login_msg]
        gr.on([login_btn.click, password.submit], sign_in, [username, password], access_outputs)
        reg.click(register, [email, name, secret], reg_msg)
        logout.click(lambda: None, outputs=token, queue=False).then(None, js="() => window.location.reload()")
        theme_btn.click(None, js=THEME_TOGGLE)
        demo.load(None, js=THEME_INIT)

        gr.on([source.input, dataset.input, problem.input, use_cutoff.input], forecast_form,
              [source, dataset, problem, use_cutoff], [dataset, file, target, use_cutoff, cutoff, hint], queue=False)
        problem.input(forecast_algorithms, problem, forecast_model, queue=False)
        gr.on([agent_provider.input, reload_models.click], ollama_options, [token, agent_provider],
              [agent_model, reload_models, agent_message])
        display = [status, result_content, open_review, cards, metrics, importance, risks, predictions, export, notes]
        serial = {"concurrency_id": "forecast-display", "show_progress": "hidden"}
        launch.click(start_forecast, [token, source, file, dataset, problem, target, use_cutoff, cutoff, iterations,
                                      agent_provider, agent_model, forecast_model],
                     [active_id, forecast_timer, form_msg, *display], **serial)
        gr.on([history_box.expand, reload_history.click], history_choices, token, [history, history_note],
              show_progress="hidden")
        history.input(lambda selected: selected, history, active_id, **serial).then(
            refresh_forecast, [token, active_id], [forecast_timer, *display], **serial)
        gr.on([forecast_timer.tick, refresh.click], refresh_forecast, [token, active_id], [forecast_timer, *display],
              **serial)
        source_in = [token, data_task, data_subset, data_page]
        source_out = [data_head, source_rows, data_notes]
        data_tab.select(history_choices, [token, data_task], [data_task, data_history_note], show_progress="hidden")
        open_source.click(open_active_data, [token, active_id],
                          [navigation, data_tabs, data_task, data_history_note, data_subset, data_page]).then(
            load_input_preview, source_in, source_out)
        data_task.input(lambda: 1, outputs=data_page, queue=False).then(load_input_preview, source_in, source_out)
        data_subset.input(lambda: 1, outputs=data_page, queue=False).then(load_input_preview, source_in, source_out)
        data_load.click(load_input_preview, source_in, source_out)
        kb_in = [token, kb_kind, kb_page]
        kb_out = [kb_summary, kb_note, kb_table]
        gr.on([kb_tab.select, kb_load.click], load_knowledge, kb_in, kb_out)
        kb_kind.input(lambda: 1, outputs=kb_page, queue=False).then(load_knowledge, kb_in, kb_out)
        open_review.click(lambda tid: (gr.update(selected="review"), tid, tid), active_id,
                          [navigation, checked_task, queue_filter], queue=False)
        use_active.click(lambda tid: tid, active_id, checked_task, queue=False)
        queue_active.click(lambda tid: tid, active_id, queue_filter, queue=False).then(
            load_queue, [token, queue_filter], [queue_table, student, queue_note])

        action.input(lambda value: (gr.update(visible=value == "choose_model"),
                                    gr.update(visible=value == "change_threshold")),
                     action, [revised_model, risk_threshold], queue=False)
        decision.input(lambda value: gr.update(visible=value == "adjust"), decision, adjusted, queue=False)
        decide.click(submit_task_hitl, [token, checked_task, action, revised_model, risk_threshold, task_comment],
                     [task_feedback, active_id, forecast_timer], **serial)
        queue_reload.click(load_queue, [token, queue_filter], [queue_table, student, queue_note])
        save_review.click(submit_review, [token, student, decision, adjusted, student_comment], review_feedback).then(
            load_queue, [token, queue_filter], [queue_table, student, queue_note])

        quality_events = {"concurrency_id": "quality-display", "show_progress": "hidden"}
        compare_out = [comparison_id, compare_msg, compare_fig, compare_scores, compare_tests]
        batch_out = [job_id, batch_timer, batch_msg, batch_fig, batch_scores]
        history_in = [token, comparison_id, job_id]
        history_out = [comparison_history, batch_history, quality_history_note]
        gr.on([quality_tab.select, refresh_quality.click], quality_history, history_in, history_out,
              **quality_events).then(open_comparison, [token, comparison_history], compare_out,
                                     **quality_events).then(open_batch, [token, batch_history], batch_out,
                                                            **quality_events)
        comparison_history.input(open_comparison, [token, comparison_history], compare_out, **quality_events)
        batch_history.input(open_batch, [token, batch_history], batch_out, **quality_events)
        compare_btn.click(compare, [token, compare_models, sample], compare_out, **quality_events).then(
            quality_history, history_in, history_out, **quality_events)
        batch_btn.click(start_batch, [token, count, modes, shares], batch_out, **quality_events).then(
            quality_history, history_in, history_out, **quality_events)
        batch_timer.tick(refresh_batch, [token, job_id], batch_out[1:], **quality_events)

        log_out = [log_head, log_lines, log_timeline, log_calls]
        open_log.click(lambda tid: (gr.update(selected="logs"), tid), active_id, [navigation, log_id], queue=False).then(
            load_logs, [token, log_id], log_out)
        log_active.click(lambda tid: tid, active_id, log_id, queue=False).then(load_logs, [token, log_id], log_out)
        logs_btn.click(load_logs, [token, log_id], log_out)
        log_live.input(lambda enabled: gr.update(active=enabled), log_live, log_timer, queue=False)
        log_timer.tick(load_logs, [token, log_id], log_out, show_progress="hidden")
        overview_btn.click(agent_overview, token, [overview, overview_status])
    return demo


if __name__ == "__main__":
    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate", spacing_size="sm",
                           font=["Inter", "Segoe UI", "system-ui"]).set(
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_text_color="#344c60", block_label_text_color_dark="#c1d1df",
        block_background_fill="#ffffff", block_background_fill_dark="#172432",
        input_background_fill="#f3f7f9", input_background_fill_dark="#202f40",
        input_border_color="#d8e4eb", input_border_color_dark="#34495c",
    )
    build_ui().launch(
        server_name=os.environ.get("MLF_UI_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("MLF_UI_PORT", "7860")),
        theme=theme, css=Path(__file__).with_name("styles.css").read_text(encoding="utf-8"),
        footer_links=[],
    )
