"""Smoke-тесты UI: сборка интерфейса и рендер дерева рассуждений."""
from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from frontend import app  # noqa: E402
from frontend.api_client import ApiClient, ApiError  # noqa: E402

NODE = {"id": "1", "agent_name": "Supervisor", "step": "run", "status": "ok", "duration_ms": 12.0, "tokens": 5,
        "tool_name": None, "output_summary": "done", "started_at": "2026-01-01T00:00:00",
        "finished_at": "2026-01-01T00:00:01",
        "children": [{"id": "2", "agent_name": "ModelSelector", "step": "tool_call", "status": "ok", "duration_ms": 3.0,
                      "tokens": 0, "tool_name": "fit_random_forest", "output_summary": "",
                      "started_at": "2026-01-01T00:00:00", "finished_at": "2026-01-01T00:00:01", "children": []}]}


def test_build_ui_and_render_helpers():
    assert isinstance(app.build_ui(), gr.Blocks)
    md = app.tree_markdown([NODE])
    assert any("fit_random_forest" in line for line in md) and "Supervisor" in md[0]
    assert app.timeline([NODE]) is not None and app.timeline([]) is None
    assert app.importance_plot({"a": 0.1}) is not None and app.importance_plot(None) is None
    assert app.risk_plot({"high": 3}) is not None and app.risk_plot({}) is None
    assert app.auto_refresh(None, "", True) == (gr.skip(),) * 6


def test_client_errors_without_server():
    c = ApiClient(base_url="http://127.0.0.1:9", timeout=1)
    with pytest.raises(Exception):  # noqa: B017 - нет сервера
        c.me()
    assert "⚠️" in app.err(ApiError("x"))
