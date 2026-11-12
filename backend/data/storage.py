"""Хранилище артефактов задачи (данные, модели, прогнозы) — в State передаются только пути."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from backend.core.config import get_settings


class ArtifactStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_settings().artifacts_dir

    def task_dir(self, task_id: str) -> Path:
        d = self.root / task_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, task_id: str, name: str) -> Path:
        return self.task_dir(task_id) / name

    def save_model(self, task_id: str, obj: Any, name: str = "model.joblib") -> str:
        p = self.path(task_id, name)
        joblib.dump(obj, p)
        return str(p)

    @staticmethod
    def load_model(path: str) -> Any:
        return joblib.load(path)

    def save_json(self, task_id: str, name: str, data: Any) -> str:
        p = self.path(task_id, name)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return str(p)

    @staticmethod
    def load_json(path: str) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))
