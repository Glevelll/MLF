# 🎓 MLF-2026 · Прогноз успеваемости студентов (задание №9)

Мультиагентная система прогноза **риска неуспеваемости/отчисления** (бинарная классификация) и **итоговой оценки**
(регрессия) по данным об активности студентов: посещаемость, сдача ДЗ, промежуточные баллы, время в LMS,
социально-демографические признаки.

- **Агенты:** `DataAnalyst → ModelSelector → Validator → Critic → Editor` под управлением `Supervisor`,
  написаны на чистом Python без LangChain/LangGraph.
- **Tool Calling:** ModelSelector и Editor не генерируют код, а вызывают инструменты из реестра
  (`fit_logistic_regression`, `fit_random_forest`, `fit_tabular_xgboost`, `fit_ridge_regression`, `fit_arima`, `fit_ets`,
  `add_features`, `drop_features`, `change_threshold`, `change_hyperparams`, `switch_model`).
  Аргументы проверяются Pydantic-схемой. Fallback: строгий function calling → разбор JSON из текста →
  повтор с уточнением (до 3 раз) → правило по умолчанию.
- **Agentic RAG:** BM25 + векторный поиск (numpy локально / Qdrant в prod), объединение через **Reciprocal Rank Fusion**,
  **Contextual Retrieval** и фильтры по метаданным (тип задачи, число признаков, доля пропусков…).
- **База знаний:** 2 844 кейса: 2 544 профиля студентов (профиль → итог + рекомендации) и 300 кейсов выбора модели.
  Справочник — 510 терминов.
- **HITL:** задача, не сошедшаяся за `max_iterations`, ждёт решения пользователя (`accept / choose_model /
  change_threshold / reject`). Студенты с высоким риском попадают в очередь преподавателя; подтверждённые
  и скорректированные решения пополняют базу знаний (active learning).
- **Observability:** иерархический Reasoning Log (`parent_id`) и журнал всех вызовов LLM: модель, провайдер,
  токены (при отсутствии — оценка через `tiktoken`), latency, статус, попадания в кэш.
- **REST API** (FastAPI, HTTP 202 + опрос), фоновые задачи (inline asyncio локально / Celery + Redis в prod),
  JWT и роли `user / editor / admin`, Gradio UI.

## Быстрый старт (локально, без Docker)

Нужен Python 3.11–3.13. Локальный режим: SQLite, in-memory векторный индекс, inline-задачи, mock-LLM.

```bash
python -m venv .venv
```
```bash
.venv\Scripts\pip install -e ".[ui,dev]"
```
```bash
.venv\Scripts\python scripts/import_data.py
```
```bash
.venv\Scripts\python scripts/seed_demo_user.py
```
```bash
.venv\Scripts\python run.py
```
В отдельном окне PowerShell из папки проекта запустите интерфейс.

```powershell
$env:NO_PROXY = '127.0.0.1,localhost'
.venv\Scripts\python -m frontend.app
```

- API и Swagger: http://127.0.0.1:8000/docs
- UI: http://127.0.0.1:7860 (демо-вход `demo` / `demo12345`, только для локальной разработки)
- Данные UCI, база знаний и справочник уже находятся в `data/`. Для пересоздания базы используйте `scripts/generate_glossary.py` и `scripts/generate_knowledge_base.py --datasets 300`, затем повторите импорт.

### Подключение настоящей LLM (Ollama)

```powershell
ollama list
ollama run qwen2.5:1.5b 'Ответь одним словом: готово'
ollama ps
```

В `.env` (шаблон — `.env.example`):
```
MLF_LLM_PROVIDER=ollama
MLF_LLM_MODEL=qwen2.5:1.5b
MLF_LLM_FALLBACK_PROVIDERS=["mock"]
```

Перезапустите локальный API после правки `.env`. Если модели ещё нет, `ollama run` скачает её при первом запуске.
Столбец `PROCESSOR` в `ollama ps` показывает, где загружена модель. На проверенном ноутбуке `qwen2.5:1.5b`
загружалась целиком на RTX 3050. Классификация и регрессия обучаются на CPU.
Запасной локальный провайдер `hf` использует Transformers. Для него установите `pip install ".[hf]"` и задайте
`MLF_LLM_PROVIDER=hf`, `MLF_HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct`. При первом запуске потребуются веса модели.
Также есть `openai` (OpenAI / Groq / OpenRouter / vLLM через `MLF_OPENAI_BASE_URL`) и `anthropic`.
Если провайдер недоступен, менеджер переходит к следующему в цепочке.

`mock` — детерминированная политика: она читает роль агента и JSON-контекст из промпта и возвращает
tool calls / JSON как настоящая модель. С ней весь пайплайн и тесты работают без GPU и сети.

## Запуск в Docker (prod)

```bash
docker compose up -d --build
```
Сервис `init` загрузит готовую базу знаний и построит индекс Qdrant. После него запустятся API, фоновые задачи и интерфейс.

В форме «Прогнозы» поле «Кто помогает агентам» переключает Mock и Ollama для конкретного задания. Список Qwen берётся
из подключённой Ollama. На Docker Desktop можно использовать уже установленные на компьютере модели, задав
`MLF_OLLAMA_URL=http://host.docker.internal:11434` в `.env` и пересоздав `app` и `worker`. При явном выборе Qwen ошибка
подключения отображается в задаче. Поле «Алгоритм прогноза» позволяет выбрать совместимую табличную модель вручную.
В режиме «Агенты выберут» алгоритм подбирает ModelSelector.

При `MLF_LLM_PROVIDER=ollama` загрузите модель, выбранную в `.env`. Для запуска без этой загрузки задайте `MLF_LLM_PROVIDER=mock`.

```bash
docker compose exec ollama ollama pull qwen2.5:7b
```

Сервисы: `app` (FastAPI), `worker` и `beat` (Celery: прогнозы, оценка, ночная переиндексация RAG, очистка кэша),
`frontend` (Gradio), `db` (PostgreSQL), `redis`, `qdrant`, `ollama`, `nginx` (проксирование, gzip, security headers;
порт 80, меняется через `MLF_HTTP_PORT`). Перед запуском задайте `MLF_JWT_SECRET` в `.env`. Схема БД: `alembic upgrade head`
(при старте таблицы также создаются автоматически).

## Использование API

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/auth/register`, `/auth/login`, GET `/auth/me` | регистрация, JWT, профиль (первый пользователь — admin) |
| POST | `/api/v1/forecast/` | файл CSV/Parquet, `data_url` или `dataset` (`uci_both`, `uci_mat`, `uci_por`, `synthetic`) + `target_column`, `problem_type`, `binarize_threshold`, `max_iterations` → **202** и `task_id` |
| GET | `/api/v1/forecast/{id}/status` | `pending / running / waiting_for_human / completed / failed` |
| GET | `/api/v1/forecast/{id}/result` | модель, метрики CV/holdout, важность признаков, оценка критика, замечания, правки Editor, RAG-кейсы, превью прогноза |
| GET | `/api/v1/forecast/{id}/predictions` | CSV: вероятность риска / прогноз оценки, уровень риска, факторы риска, рекомендации, похожие студенты |
| GET | `/api/v1/forecast/{id}/reasoning` | дерево рассуждений и вызовы LLM |
| POST | `/api/v1/forecast/{id}/hitl` | решение по задаче: `accept`, `choose_model`, `change_threshold`, `reject` |
| GET/POST | `/api/v1/hitl/queue`, `/api/v1/hitl/reviews/{id}` | очередь студентов высокого риска: `confirm / adjust / reject` (роль editor) |
| GET | `/api/v1/evaluate/compare?models=...&sample_size=...` | CV-сравнение моделей + Wilcoxon / t-test / Диболд–Мариано |
| POST/GET | `/api/v1/evaluate/batch`, `/api/v1/evaluate/batch/{id}`, `/api/v1/evaluate/results` | пакетная оценка режимов |
| GET | `/api/v1/logs/llm-calls`, `/api/v1/logs/agent-metrics`, `/api/v1/kb/stats`, `/api/v1/kb/search` | наблюдаемость и база знаний |

Строки с пустой целевой колонкой считаются «новыми студентами»: для них строится прогноз финальной моделью.
Для размеченных строк выдаются out-of-fold прогнозы, полученные без утечек.

## Архитектура

```text
backend/
├── agents/        base.py (Agent, AgentContext), data_analyst.py, model_selector.py, validator.py,
│                  critic.py (LLM-as-a-Judge), editor.py, supervisor.py (цикл, HITL, финализация), common.py
├── api/           main.py, deps.py (JWT, роли), schemas.py, routes/{auth,forecast,hitl,evaluate,system}.py
├── core/          config.py (pydantic-settings), models.py (ForecastState), metrics.py
│   └── llm/       manager.py (роутинг, retry, кэш, логирование, call_tool/structured), tools.py (реестр),
│                  parsing.py (fallback-парсер), tokens.py, providers/{mock,ollama,hf_local,api}.py
├── tabular/       preprocessing.py, features.py (авто feature engineering), training.py, datasets.py
├── timeseries/    preprocessing.py (ресемплинг, ADF, STL), classical.py (ARIMA, ETS) — точка расширения
├── data/          embeddings.py (FeatureExtractor: tabular/timeseries, LSA, CaseEmbedder 128d),
│                  knowledge_base.py, cache.py (DB/Redis), storage.py (артефакты по ссылке)
├── retrieval/     bm25_index.py, vector_index.py (memory/Qdrant), hybrid_retriever.py (RRF), index_manager.py
├── evaluation/    specs.py, runner.py (режимы, compare_models), stats.py (DM, Wilcoxon, t-test, bootstrap)
├── hitl/          active_learning.py, feedback.py
├── db/            models.py, session.py, migrations/ (Alembic)
├── utils/         tracing.py (Reasoning Log), logging.py, security.py
└── worker/        pipeline.py, jobs.py, dispatcher.py (inline/Celery), celery_app.py (+ beat)
frontend/          app.py (Gradio), api_client.py
scripts/           download_data, generate_glossary, generate_knowledge_base, import_data, seed_demo_user,
                   run_eval, make_report_plots
tests/             pytest + pytest-asyncio + respx (84 теста)
```

**Цикл Supervisor.** `DataAnalyst` запускается один раз. Затем до `max_iterations` раз выполняется
`ModelSelector → Validator → Critic`. Цикл сходится, если Validator не выдал блокирующих замечаний и
оценка Critic ≥ `critic_threshold` (по умолчанию 6). Иначе `Editor` вносит правки (признаки, порог,
гиперпараметры, смена модели) и начинается новая итерация. Если цикл не сошёлся или есть критические замечания,
задача получает статус `waiting_for_human`.

**State** (`ForecastState`) строго типизирован. DataFrame и модели в нём не хранятся, только пути к артефактам
(`data/artifacts/<task_id>/`). Препроцессинг (импутация, OHE, масштабирование) встроен в sklearn `Pipeline`
и подгоняется внутри фолдов CV, поэтому утечек нет.

**Проверки Validator:** форма прогнозов и NaN, диапазон вероятностей, утечка целевой переменной
(одиночный признак с AUC/|r| ≥ 0.97), доминирующий признак, переобучение (разрыв train/test), калибровка (ECE),
recall группы риска, порог качества (ROC-AUC 0.7 / R² 0.3), нормальность остатков (Shapiro–Wilk);
для рядов — Ljung–Box.

**Интерпретируемость.** Глобально: permutation importance на holdout. Локально: для каждого студента группы риска
считается occlusion-эффект (как изменится риск, если заменить значение признака на типичное). Факторы риска
объединяются с рекомендациями похожих студентов из RAG.

## Данные

| Датасет | Тип | Строк | Признаков |
|---|---|---|---|
| UCI Student Performance (Cortez & Silva, 2008), math + portuguese | реальный | 1 044 | 32 (+ цель G3); варианты без G1/G2 |
| OULAD-подобный синтетический генератор (`backend/tabular/datasets.py`) | синтетика | 200–4 000 | 20 (+0–15 шумовых) |

Цели: `G3 < 10` (неуспеваемость), `G3 < 12`, `G3` (регрессия), `dropout`, `final_score < 40`, `final_score`.
Спецификации наборов для базы знаний (`split=kb`) и для оценки (`split=eval`, 200 спецификаций) используют
непересекающиеся seed-ы. Реальные данные OULAD / Kaggle можно скачать через `scripts/download_data.py --kaggle`.

## Оценка качества

```bash
.venv\Scripts\python scripts/run_eval.py --num-series 30 --modes baseline +rag +agents --data-fractions 0.1 0.2 0.5 1.0
```
```bash
.venv\Scripts\python scripts/make_report_plots.py
```

Режимы: `baseline` (логистическая / Ridge-регрессия без LLM), `+rag` (голосование лучших моделей похожих кейсов),
`+agents` (полный цикл), `+foundation` (TabPFN, если установлен). Метрики: ROC-AUC, F1, Accuracy, Recall, Brier, ECE;
для регрессии — MAE, RMSE, R², MAPE, WAPE. Статистика: парные Wilcoxon и t-test по наборам, bootstrap 95% CI
прироста, тест Диболда–Мариано на поэлементных потерях. Агентные метрики: итерации, доля HITL, время, токены.
Для табличной задачи `--horizons 10 20 50` означает размер отложенной выборки «новых студентов» в процентах.
Результаты сохраняются в `results/eval_*/` (`rows.csv`, `summary.json`, `report.md`), лог — в `eval.log`,
сводки — в БД. Анализ результатов — в [REPORT.md](REPORT.md).

Для UCI записи одного ученика группируются по правилам сопоставления из исходного набора. В кросс-валидации и при выделении теста группа целиком попадает в одну часть. У `uci_both` 1 044 строки по двум предметам, часть учеников встречается дважды. Фактическая доля отложенных строк может немного отличаться от заданной, поскольку деление идёт по группам.

Пакетная оценка через скрипт и API строит отдельный RAG-индекс из синтетических кейсов выбора модели. Она исключает кейсы с seed, использованным в проверяемом синтетическом наборе. Для режима `baseline` индекс не требуется. В обычном прогнозе используется рабочая база знаний. В отчёте `+agents` поле `llm_models_by_run` показывает, какие провайдеры и модели ответили на самом деле.

## Тесты

```bash
.venv\Scripts\python -m pytest --cov=backend
```

87 тестов, покрытие **92,14%** в проверенном запуске на Windows. Покрыты: юнит-тесты каждого агента (с мок-LLM), Tool Calling и fallback (3 попытки →
правило по умолчанию), провайдеры Ollama / OpenAI / Anthropic через `respx`, предобработка таблиц и рядов,
BM25 / вектор / RRF / Qdrant (local mode), интеграционные тесты API (полный цикл, HITL, роли, ошибки),
оценка и статистика, скрипты. CI — `.github/workflows/ci.yml` (ruff, pytest с порогом покрытия 70%, сборка Docker).
