"""Интеграционные тесты API (httpx + ASGITransport, inline-фоновые задачи, mock LLM)."""
from __future__ import annotations

import io

import httpx
import numpy as np
import pandas as pd
import pytest

from backend.api.main import create_app
from backend.db.session import init_db
from backend.tabular.datasets import generate_synthetic
from backend.worker.dispatcher import wait_all
from tests.conftest import needs_uci


@pytest.fixture
async def client():
    await init_db()
    app = create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=600) as c:
        yield c


async def auth(client, name: str) -> dict:
    r = await client.post("/auth/register", json={"email": f"{name}@uni.ru", "username": name, "password": "secret123"})
    assert r.status_code in (201, 409)
    r = await client.post("/auth/login", data={"username": name, "password": "secret123"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


async def test_health_and_auth_flow(client):
    assert (await client.get("/healthz")).json()["status"] == "ok"
    admin = await auth(client, "admin0")  # первый пользователь -> admin
    me = (await client.get("/auth/me", headers=admin)).json()
    assert me["role"] == "admin"
    assert (await client.post("/auth/register", json={"email": "admin0@uni.ru", "username": "admin0",
                                                      "password": "secret123"})).status_code == 409
    assert (await client.post("/auth/login", data={"username": "admin0", "password": "wrong"})).status_code == 401
    assert (await client.get("/auth/me")).status_code == 401
    assert (await client.get("/auth/me", headers={"Authorization": "Bearer junk"})).status_code == 401
    user = await auth(client, "student1")
    uid = (await client.get("/auth/me", headers=user)).json()["id"]
    assert (await client.post(f"/auth/users/{uid}/role?role=editor", headers=user)).status_code == 403
    r = await client.post(f"/auth/users/{uid}/role?role=editor", headers=admin)
    assert r.json()["role"] == "editor"
    assert (await client.post(f"/auth/users/{uid}/role?role=god", headers=admin)).status_code == 400
    assert (await client.post("/auth/users/9999/role?role=user", headers=admin)).status_code == 404


async def test_forecast_full_cycle_with_hitl_reviews(client):
    await auth(client, "admin0")
    teacher = await auth(client, "teacher")
    df = generate_synthetic(330, seed=21).drop(columns=["final_score"])
    df.loc[:14, "dropout"] = np.nan
    r = await client.post("/api/v1/forecast/", headers=teacher,
                          files={"file": ("students.csv", csv_bytes(df), "text/csv")},
                          data={"target_column": "dropout"})
    assert r.status_code == 202, r.text
    task_id = r.json()["task_id"]
    await wait_all()
    st = (await client.get(f"/api/v1/forecast/{task_id}/status", headers=teacher)).json()
    assert st["status"] in ("completed", "waiting_for_human"), st
    if st["status"] == "waiting_for_human":
        r = await client.post(f"/api/v1/forecast/{task_id}/hitl", headers=teacher, json={"action": "accept"})
        assert r.status_code == 202
        await wait_all()
    res = (await client.get(f"/api/v1/forecast/{task_id}/result", headers=teacher)).json()
    assert res["status"] == "completed" and res["model_type"] and res["metrics"]["cv"]["roc_auc"] > 0.6
    assert res["predictions_preview"] and res["available_models"]
    tree = (await client.get(f"/api/v1/forecast/{task_id}/reasoning", headers=teacher)).json()
    assert tree["n_spans"] >= 5 and tree["tree"][0]["agent_name"] == "Supervisor" and tree["llm_calls"]
    dl = await client.get(f"/api/v1/forecast/{task_id}/predictions", headers=teacher)
    assert dl.status_code == 200 and b"risk_probability" in dl.content
    tasks = (await client.get("/api/v1/forecast/", headers=teacher)).json()
    listed = next(t for t in tasks if t["task_id"] == task_id)
    assert listed["source"] == "students.csv" and listed["target_column"] == "dropout"
    data_path = f"/api/v1/forecast/{task_id}/data"
    source = (await client.get(data_path, headers=teacher, params={"subset": "new", "offset": 2, "limit": 3})).json()
    assert source["source"] == "students.csv" and source["known_rows"] == 315 and source["new_rows"] == 15
    assert source["matching_rows"] == 15 and len(source["rows"]) == 3 and "dropout" not in source["features"]
    assert all(row[source["columns"].index("dropout")] is None for row in source["rows"])

    # другой пользователь не видит задачу
    other = await auth(client, "other")
    assert (await client.get(f"/api/v1/forecast/{task_id}/status", headers=other)).status_code == 403
    assert (await client.get(data_path, headers=other)).status_code == 403
    assert (await client.get("/api/v1/kb/records", headers=other)).status_code == 403
    assert (await client.get("/api/v1/hitl/queue", headers=other)).status_code == 403

    # HITL по студентам: роль editor
    admin = await auth(client, "admin0")
    q = (await client.get(f"/api/v1/hitl/queue?kind=student_risk&task_id={task_id}", headers=admin)).json()
    if q:
        rid = q[0]["id"]
        r = await client.post(f"/api/v1/hitl/reviews/{rid}", headers=admin,
                              json={"decision": "adjust", "adjusted_risk": 0.9, "comment": "семейные обстоятельства"})
        assert r.status_code == 200 and r.json()["kb_added"] and r.json()["final_risk"] == 0.9
        assert (await client.post(f"/api/v1/hitl/reviews/{rid}", headers=admin,
                                  json={"decision": "confirm"})).status_code == 409
        if len(q) > 1:
            r = await client.post(f"/api/v1/hitl/reviews/{q[1]['id']}", headers=admin, json={"decision": "reject"})
            assert r.json()["status"] == "rejected" and not r.json()["kb_added"]
    assert (await client.post("/api/v1/hitl/reviews/99999", headers=admin, json={"decision": "confirm"})).status_code == 404
    stats = (await client.get("/api/v1/hitl/stats", headers=admin)).json()
    assert isinstance(stats, dict)
    kb = (await client.get("/api/v1/kb/stats", headers=admin)).json()
    assert "glossary" in kb
    records = (await client.get("/api/v1/kb/records?kind=student_case&limit=1", headers=admin)).json()
    assert records["total"] == kb["knowledge_base"].get("student_case", 0) and len(records["rows"]) <= 1
    am = (await client.get("/api/v1/logs/agent-metrics", headers=admin)).json()
    assert am["tasks_finished"] >= 1 and am["llm_calls"] >= 1
    calls = (await client.get(f"/api/v1/logs/llm-calls?request_id={task_id}&agent=Critic", headers=admin)).json()
    assert all(c["agent"] == "Critic" for c in calls)


async def test_forecast_hitl_task_quality_and_choose_model(client):
    user = await auth(client, "hitluser")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=160), "b": rng.normal(size=160), "y": rng.integers(0, 2, 160)})
    r = await client.post("/api/v1/forecast/", headers=user, files={"file": ("n.csv", csv_bytes(df), "text/csv")},
                          data={"target_column": "y", "max_iterations": "2"})
    tid = r.json()["task_id"]
    await wait_all()
    st = (await client.get(f"/api/v1/forecast/{tid}/status", headers=user)).json()
    assert st["status"] == "waiting_for_human" and st["hitl_reason"]
    res = (await client.get(f"/api/v1/forecast/{tid}/result", headers=user)).json()
    assert res["status"] == "waiting_for_human"
    bad = await client.post(f"/api/v1/forecast/{tid}/hitl", headers=user, json={"action": "choose_model",
                                                                                 "model_type": "fit_arima"})
    assert bad.status_code == 400
    assert (await client.post(f"/api/v1/forecast/{tid}/hitl", headers=user,
                              json={"action": "change_threshold"})).status_code == 400
    assert (await client.post(f"/api/v1/forecast/{tid}/hitl", headers=user,
                              json={"action": "change_horizon"})).status_code == 400
    r = await client.post(f"/api/v1/forecast/{tid}/hitl", headers=user,
                          json={"action": "choose_model", "model_type": "fit_random_forest"})
    assert r.status_code == 202
    await wait_all()
    res = (await client.get(f"/api/v1/forecast/{tid}/result", headers=user)).json()
    assert res["status"] == "completed" and res["model_type"] == "fit_random_forest"
    assert res["hitl_resolution"]["action"] == "choose_model"
    assert (await client.post(f"/api/v1/forecast/{tid}/hitl", headers=user, json={"action": "accept"})).status_code == 409


async def test_forecast_reject_and_errors(client):
    user = await auth(client, "erruser")
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(size=120), "y": rng.integers(0, 2, 120)})
    r = await client.post("/api/v1/forecast/", headers=user, files={"file": ("n.csv", csv_bytes(df), "text/csv")},
                          data={"target_column": "y", "max_iterations": "1"})
    tid = r.json()["task_id"]
    await wait_all()
    if (await client.get(f"/api/v1/forecast/{tid}/status", headers=user)).json()["status"] == "waiting_for_human":
        await client.post(f"/api/v1/forecast/{tid}/hitl", headers=user, json={"action": "reject"})
        await wait_all()
        assert (await client.get(f"/api/v1/forecast/{tid}/status", headers=user)).json()["status"] == "failed"
        assert (await client.get(f"/api/v1/forecast/{tid}/result", headers=user)).status_code == 409
    # ошибочные запросы
    assert (await client.post("/api/v1/forecast/", headers=user, data={})).status_code == 400
    assert (await client.post("/api/v1/forecast/", headers=user, data={"dataset": "nope"})).status_code == 400
    assert (await client.post("/api/v1/forecast/", headers=user, data={"data_url": "ftp://x"})).status_code == 400
    assert (await client.post("/api/v1/forecast/", headers=user,
                              files={"file": ("x.exe", b"MZ", "application/octet-stream")})).status_code == 400
    assert (await client.post("/api/v1/forecast/", headers=user,
                              files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})).status_code == 400
    assert (await client.get("/api/v1/forecast/nope/status", headers=user)).status_code == 404
    assert (await client.get(f"/api/v1/forecast/{tid}/predictions", headers=user)).status_code in (200, 404)
    # неверная целевая колонка -> задача падает с ошибкой
    r = await client.post("/api/v1/forecast/", headers=user, files={"file": ("n.csv", csv_bytes(df), "text/csv")},
                          data={"target_column": "zzz"})
    await wait_all()
    st = (await client.get(f"/api/v1/forecast/{r.json()['task_id']}/status", headers=user)).json()
    assert st["status"] == "failed" and "zzz" in st["error"]
    failed = (await client.get("/api/v1/forecast/?status_filter=failed", headers=user)).json()
    assert failed


@needs_uci
@pytest.mark.usefixtures("evaluation_kb")
async def test_builtin_dataset_and_evaluate(client):
    user = await auth(client, "evaluser")
    r = await client.post("/api/v1/forecast/", headers=user,
                          data={"dataset": "uci_mat", "binarize_threshold": "10", "max_iterations": "1"})
    assert r.status_code == 202
    assert (await client.post("/api/v1/forecast/", headers=user,
                              data={"dataset": "uci_mat", "target_column": "nope"})).status_code == 400
    cmp_ = await client.get("/api/v1/evaluate/compare?models=fit_logistic_regression,fit_random_forest"
                            "&sample_size=200", headers=user)
    assert cmp_.status_code == 200 and cmp_.json()["winner"] in ("fit_logistic_regression", "fit_random_forest")
    assert len(cmp_.json()["tests"]) == 1 and "diebold_mariano" in cmp_.json()["tests"][0]
    assert (await client.get("/api/v1/evaluate/compare?models=fit_magic,fit_random_forest",
                             headers=user)).status_code == 400
    assert (await client.get("/api/v1/evaluate/compare?models=fit_ridge_regression,fit_random_forest",
                             headers=user)).status_code == 400
    assert (await client.get("/api/v1/evaluate/compare?dataset=x", headers=user)).status_code == 400
    b = await client.post("/api/v1/evaluate/batch", headers=user,
                          json={"num_datasets": 2, "modes": ["baseline", "+rag"], "fractions": [0.5, 1.0]})
    assert b.status_code == 202
    await wait_all()
    job = (await client.get(f"/api/v1/evaluate/batch/{b.json()['task_id']}", headers=user)).json()
    assert job["status"] == "completed" and job["summary"]["groups"]
    assert (await client.get("/api/v1/evaluate/batch/nope", headers=user)).status_code == 404
    history = (await client.get("/api/v1/evaluate/history", headers=user)).json()
    comparison_id = cmp_.json()["comparison_id"]
    assert history["comparisons"][0]["id"] == comparison_id
    assert history["batches"][0]["id"] == job["job_id"] and history["batches"][0]["status"] == "completed"
    saved = await client.get(f"/api/v1/evaluate/compare/{comparison_id}", headers=user)
    assert saved.status_code == 200 and saved.json() == cmp_.json()
    assert (await client.get("/api/v1/evaluate/compare/999999", headers=user)).status_code == 404
    other = await auth(client, "other-evaluser")
    assert (await client.get("/api/v1/evaluate/history", headers=other)).json()["batches"] == []
    res = (await client.get("/api/v1/evaluate/results", headers=user)).json()
    assert res["comparisons"] and res["evaluations"]
    s = (await client.get("/api/v1/kb/search?q=пропуски", headers=user)).json()
    assert "results" in s


async def test_admin_reindex(client):
    admin = await auth(client, "admin0")
    r = await client.post("/api/v1/admin/reindex", headers=admin)
    assert r.status_code == 202
    await wait_all()
