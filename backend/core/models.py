"""Pydantic-модели домена: состояние пайплайна (State), замечания, статусы."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    pending = "pending"
    running = "running"
    waiting_for_human = "waiting_for_human"
    completed = "completed"
    failed = "failed"


class Issue(BaseModel):
    code: str
    severity: Literal["critical", "warning", "info"] = "warning"
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class EditorAction(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ForecastState(BaseModel):
    """Состояние, передаваемое между агентами.

    Тяжёлые объекты (DataFrame, модели) не хранятся в State — только пути к артефактам.
    """

    task_id: str
    series_id: str  # идентификатор набора данных (для табличных задач — датасет студентов)
    horizon: int = 0  # для временных рядов — горизонт; для табличных не используется
    freq: str | None = None
    task_type: Literal["timeseries", "tabular"] = "tabular"
    metadata: dict[str, Any] = Field(default_factory=dict)
    target_column: str | None = None
    problem_type: Literal["auto", "classification", "regression"] = "auto"
    # классификация по порогу: target < threshold -> положительный класс «в группе риска»
    binarize_threshold: float | None = None
    positive_label: str | None = None
    id_column: str | None = None

    # ссылки на артефакты
    data_path: str | None = None
    processed_path: str | None = None
    model_path: str | None = None
    oof_path: str | None = None
    predictions_path: str | None = None
    summary_path: str | None = None

    # управление признаками и моделью
    feature_columns: list[str] = Field(default_factory=list)
    dropped_features: list[str] = Field(default_factory=list)
    feature_ops: list[str] = Field(default_factory=list)
    decision_threshold: float = 0.5
    model_type: str | None = None
    model_params: dict[str, Any] | None = None
    forced_model: str | None = None
    selection_mode: str | None = None  # native | text | default | forced | carried

    metrics: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    validation_issues: list[Issue] = Field(default_factory=list)
    validation_passed: bool = False
    critic_score: float | None = None
    critic_reasoning: str | None = None
    critic_details: dict[str, Any] | None = None
    editor_changed: bool = False
    editor_actions: list[EditorAction] = Field(default_factory=list)
    rag_context: list[dict[str, Any]] = Field(default_factory=list)

    iteration: int = 0
    max_iterations: int = 3
    log_id: str | None = None
    hitl_requested: bool = False
    hitl_reason: str | None = None
    hitl_resolution: dict[str, Any] | None = None
    tokens_used: int = 0

    def critical_issues(self) -> list[Issue]:
        return [i for i in self.validation_issues if i.severity == "critical"]
