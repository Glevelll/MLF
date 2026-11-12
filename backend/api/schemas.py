"""Схемы запросов/ответов API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=72)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    username: str
    role: str
    is_active: bool
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class TaskAccepted(BaseModel):
    task_id: str
    status: str
    execution: str


class TaskStatusOut(BaseModel):
    task_id: str
    status: str
    task_type: str
    created_at: datetime
    updated_at: datetime
    iteration: int | None = None
    hitl_reason: str | None = None
    error: str | None = None


class HitlDecision(BaseModel):
    action: Literal["accept", "choose_model", "change_threshold", "change_horizon", "reject"]
    model_type: str | None = None
    params: dict[str, Any] | None = None
    threshold: float | None = Field(None, ge=0.05, le=0.95)
    horizon: int | None = Field(None, ge=1)
    comment: str = ""


class ReviewDecision(BaseModel):
    decision: Literal["confirm", "adjust", "reject"]
    adjusted_risk: float | None = Field(None, ge=0, le=1)
    comment: str = ""
    recommendations: list[str] | None = None


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    task_id: str
    kind: str
    status: str
    priority: float
    payload: dict[str, Any]
    decision: dict[str, Any] | None
    created_at: datetime


class BatchEvalRequest(BaseModel):
    num_datasets: int = Field(5, ge=1, le=200)
    modes: list[Literal["baseline", "+rag", "+agents", "+foundation"]] = ["baseline", "+rag"]
    fractions: list[float] = [1.0]
    holdouts: list[float] = [0.2]
    seed: int = 0
