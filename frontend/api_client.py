"""Синхронный клиент FastAPI для UI."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

API_URL = os.environ.get("MLF_API_URL", "http://127.0.0.1:8000")


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url: str = API_URL, token: str | None = None, timeout: float = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _h(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _req(self, method: str, url: str, **kw) -> Any:
        r = self.http.request(method, url, headers=self._h(), **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(f"{r.status_code}: {detail}")
        ctype = r.headers.get("content-type", "")
        return r.json() if "json" in ctype else r.content

    # --- auth
    def register(self, email: str, username: str, password: str) -> dict:
        return self._req("POST", "/auth/register", json={"email": email, "username": username, "password": password})

    def login(self, username: str, password: str) -> dict:
        data = self._req("POST", "/auth/login", data={"username": username, "password": password})
        self.token = data["access_token"]
        return data

    def me(self) -> dict:
        return self._req("GET", "/auth/me")

    # --- forecast
    def create_forecast(self, file_path: str | None = None, **params: Any) -> dict:
        data = {k: str(v) for k, v in params.items() if v not in (None, "")}
        if file_path:
            with open(file_path, "rb") as f:
                return self._req("POST", "/api/v1/forecast/", data=data, files={"file": (Path(file_path).name, f)})
        return self._req("POST", "/api/v1/forecast/", data=data)

    def tasks(self) -> list[dict]:
        return self._req("GET", "/api/v1/forecast/")

    def status(self, task_id: str) -> dict:
        return self._req("GET", f"/api/v1/forecast/{task_id}/status")

    def result(self, task_id: str, preview: int = 50) -> dict:
        return self._req("GET", f"/api/v1/forecast/{task_id}/result", params={"preview": preview})

    def reasoning(self, task_id: str) -> dict:
        return self._req("GET", f"/api/v1/forecast/{task_id}/reasoning")

    def download_predictions(self, task_id: str) -> str:
        content = self._req("GET", f"/api/v1/forecast/{task_id}/predictions")
        path = Path(tempfile.gettempdir()) / f"predictions_{task_id}.csv"
        path.write_bytes(content)
        return str(path)

    def hitl(self, task_id: str, **decision: Any) -> dict:
        return self._req("POST", f"/api/v1/forecast/{task_id}/hitl",
                         json={k: v for k, v in decision.items() if v not in (None, "")})

    # --- HITL очередь
    def queue(self, kind: str | None = None, task_id: str | None = None) -> list[dict]:
        params = {k: v for k, v in {"kind": kind, "task_id": task_id}.items() if v}
        return self._req("GET", "/api/v1/hitl/queue", params=params)

    def review(self, review_id: int, decision: str, adjusted_risk: float | None = None, comment: str = "") -> dict:
        body = {"decision": decision, "comment": comment}
        if adjusted_risk is not None:
            body["adjusted_risk"] = adjusted_risk
        return self._req("POST", f"/api/v1/hitl/reviews/{review_id}", json=body)

    # --- оценка
    def compare(self, models: list[str], sample_size: int | None = None) -> dict:
        params: dict[str, Any] = {"models": ",".join(models)}
        if sample_size:
            params["sample_size"] = sample_size
        return self._req("GET", "/api/v1/evaluate/compare", params=params)

    def batch(self, **params: Any) -> dict:
        return self._req("POST", "/api/v1/evaluate/batch", json=params)

    def batch_status(self, job_id: str) -> dict:
        return self._req("GET", f"/api/v1/evaluate/batch/{job_id}")

    def results(self) -> dict:
        return self._req("GET", "/api/v1/evaluate/results")

    # --- логи
    def llm_calls(self, request_id: str | None = None) -> list[dict]:
        return self._req("GET", "/api/v1/logs/llm-calls", params={"request_id": request_id} if request_id else {})

    def agent_metrics(self) -> dict:
        return self._req("GET", "/api/v1/logs/agent-metrics")

    def kb_stats(self) -> dict:
        return self._req("GET", "/api/v1/kb/stats")
