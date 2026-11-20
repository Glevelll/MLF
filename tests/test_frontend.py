"""Smoke-тесты UI: сборка интерфейса и рендер дерева рассуждений."""
from __future__ import annotations

import httpx
import pytest
import respx

gr = pytest.importorskip("gradio")

from frontend import app  # noqa: E402
from frontend.api_client import ApiClient, ApiError  # noqa: E402
from frontend.presentation import importance_plot, risk_plot, timeline, tree_markdown  # noqa: E402

NODE = {"id": "1", "agent_name": "Supervisor", "step": "run", "status": "ok", "duration_ms": 12.0, "tokens": 5,
        "tool_name": None, "output_summary": "done", "started_at": "2026-01-01T00:00:00",
        "finished_at": "2026-01-01T00:00:01",
        "children": [{"id": "2", "agent_name": "ModelSelector", "step": "tool_call", "status": "ok", "duration_ms": 3.0,
                      "tokens": 0, "tool_name": "fit_random_forest", "output_summary": "",
                      "started_at": "2026-01-01T00:00:00", "finished_at": "2026-01-01T00:00:01", "children": []}]}


def test_build_ui_and_render_helpers():
    assert isinstance(app.build_ui(), gr.Blocks)
    md = tree_markdown([NODE])
    assert any("fit_random_forest" in line for line in md) and "Supervisor" in md[0]
    assert timeline([NODE]) is not None and timeline([]) is None
    assert importance_plot({"a": 0.1}) is not None and importance_plot(None) is None
    assert risk_plot({"high": 3}) is not None and risk_plot({}) is None
    assert app.refresh_forecast(None, "") == (gr.skip(),) * 11


def test_forecast_defaults_match_selected_task():
    def payload(source, dataset, problem, file=None, target=None):
        return app.forecast_payload(source, file, dataset, problem, target, False, 10, 3)

    assert payload("builtin", "uci_mat", "classification")["binarize_threshold"] == 10
    grade = payload("builtin", "synthetic", "regression")
    assert grade["target_column"] == "final_score" and "binarize_threshold" not in grade
    risk = payload("builtin", "synthetic", "classification")
    assert risk["target_column"] == "dropout" and "binarize_threshold" not in risk
    uploaded = payload("file", "uci_mat", "regression", "data.csv", " score ")
    assert uploaded["target_column"] == "score" and "dataset" not in uploaded
    with pytest.raises(ValueError, match="Загрузите файл"):
        payload("file", "uci_mat", "classification", "data.csv")


def test_client_errors_without_server():
    base_url = "http://127.0.0.1:9"
    c = ApiClient(base_url=base_url, timeout=1)
    assert c.trust_env is False
    with respx.mock as router:
        router.get(f"{base_url}/auth/me").mock(side_effect=httpx.ConnectError("offline"))
        with pytest.raises(ApiError, match="Не удалось подключиться к серверу"):
            c.me()
    assert "&lt;form&gt;" in app.err(ApiError("<form>"))
