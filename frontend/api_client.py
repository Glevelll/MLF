"""Синхронный клиент FastAPI для UI."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

API_URL = os.environ.get("MLF_API_URL", "http://127.0.0.1:8000")


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ApiClient:
    def __init__(self, base_url: str = API_URL, token: str | None = None, timeout: float = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.trust_env = httpx.URL(self.base_url).host not in {"localhost", "127.0.0.1", "::1", "app"}

    def _h(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _req(self, method: str, url: str, **kw) -> Any:
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout, trust_env=self.trust_env) as http:
                r = http.request(method, url, headers=self._h(), **kw)
        except httpx.TimeoutException as exc:
            raise ApiError("Сервер долго отвечает. Попробуйте ещё раз.") from exc
        except httpx.RequestError as exc:
            raise ApiError("Не удалось подключиться к серверу. Проверьте, что MLF запущен.") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            if r.status_code >= 500:
                detail = "На сервере произошла ошибка. Попробуйте ещё раз."
            elif not isinstance(detail, str):
                detail = {401: "Войдите в систему.", 403: "Недостаточно прав.",
                          413: "Файл больше 50 МБ.", 422: "Проверьте значения в форме."}.get(
                              r.status_code, "Не удалось выполнить запрос.")
            raise ApiError(detail, r.status_code)
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
    def agent_models(self) -> dict:
        return self._req("GET", "/api/v1/forecast/agent-models")

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

    def input_data(self, task_id: str, subset: str, offset: int, limit: int) -> dict:
        return self._req("GET", f"/api/v1/forecast/{task_id}/data",
                         params={"subset": subset, "offset": offset, "limit": limit})

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

    def evaluation_history(self) -> dict:
        return self._req("GET", "/api/v1/evaluate/history")

    def saved_comparison(self, comparison_id: int) -> dict:
        return self._req("GET", f"/api/v1/evaluate/compare/{comparison_id}")

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

    def kb_records(self, kind: str, offset: int, limit: int) -> dict:
        return self._req("GET", "/api/v1/kb/records", params={"kind": kind, "offset": offset, "limit": limit})
