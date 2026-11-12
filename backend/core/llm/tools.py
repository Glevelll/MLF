"""Реестр инструментов для Tool Calling.

LLM не генерирует код: она возвращает имя инструмента и аргументы (гиперпараметры),
которые валидируются Pydantic-схемой и исполняются детерминированным кодом.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.tabular.features import AVAILABLE_OPS


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ модели (табличные)
class LogisticRegressionArgs(_Args):
    C: float = Field(1.0, gt=0, le=100, description="обратная сила регуляризации")
    l1_ratio: float = Field(0.0, ge=0, le=1, description="0 = L2, 1 = L1, между — ElasticNet")
    class_weight: Literal["balanced"] | None = Field("balanced", description="балансировка классов")
    max_iter: int = Field(3000, ge=100, le=20000)


class RidgeArgs(_Args):
    alpha: float = Field(1.0, gt=0, le=1000, description="сила L2-регуляризации")


class RandomForestArgs(_Args):
    n_estimators: int = Field(300, ge=50, le=1000)
    max_depth: int | None = Field(None, ge=2, le=50)
    min_samples_leaf: int = Field(2, ge=1, le=50)
    max_features: Literal["sqrt", "log2"] | float = Field("sqrt")
    class_weight: Literal["balanced", "balanced_subsample"] | None = Field(None, description="только классификация")


class XGBoostArgs(_Args):
    n_estimators: int = Field(300, ge=50, le=1500)
    max_depth: int = Field(4, ge=2, le=12)
    learning_rate: float = Field(0.05, ge=0.005, le=0.5)
    subsample: float = Field(0.9, ge=0.4, le=1.0)
    colsample_bytree: float = Field(0.8, ge=0.3, le=1.0)
    min_child_weight: float = Field(1.0, ge=0, le=20)
    reg_lambda: float = Field(1.0, ge=0, le=50)
    scale_pos_weight: float | None = Field(None, gt=0, le=100, description="вес положительного класса")


# ------------------------------------------------------------------ модели (временные ряды)
class ArimaArgs(_Args):
    order: tuple[int, int, int] = (1, 1, 1)
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)


class EtsArgs(_Args):
    trend: Literal["add", "mul"] | None = "add"
    seasonal: Literal["add", "mul"] | None = None
    seasonal_periods: int | None = Field(None, ge=2, le=366)


# ------------------------------------------------------------------ инструменты Editor-а
class AddFeaturesArgs(_Args):
    ops: list[Literal[tuple(AVAILABLE_OPS)]] = Field(..., min_length=1)  # type: ignore[valid-type]


class DropFeaturesArgs(_Args):
    columns: list[str] = Field(..., min_length=1)
    reason: str = ""


class ChangeThresholdArgs(_Args):
    threshold: float = Field(..., ge=0.05, le=0.95)


class ChangeHyperparamsArgs(_Args):
    params: dict[str, Any]


class SwitchModelArgs(_Args):
    model_type: str


def _build_logreg(a: dict, problem_type: str, seed: int):
    from sklearn.linear_model import LogisticRegression

    solver = "saga" if a["l1_ratio"] > 0 else "lbfgs"
    return LogisticRegression(C=a["C"], l1_ratio=a["l1_ratio"], class_weight=a["class_weight"],
                              max_iter=a["max_iter"], solver=solver, random_state=seed)


def _build_ridge(a: dict, problem_type: str, seed: int):
    from sklearn.linear_model import Ridge

    return Ridge(alpha=a["alpha"])


def _build_rf(a: dict, problem_type: str, seed: int):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    common = dict(n_estimators=a["n_estimators"], max_depth=a["max_depth"], min_samples_leaf=a["min_samples_leaf"],
                  max_features=a["max_features"], random_state=seed, n_jobs=-1)
    if problem_type == "classification":
        return RandomForestClassifier(class_weight=a["class_weight"], **common)
    return RandomForestRegressor(**common)


def _build_xgb(a: dict, problem_type: str, seed: int):
    from xgboost import XGBClassifier, XGBRegressor

    common = dict(n_estimators=a["n_estimators"], max_depth=a["max_depth"], learning_rate=a["learning_rate"],
                  subsample=a["subsample"], colsample_bytree=a["colsample_bytree"],
                  min_child_weight=a["min_child_weight"], reg_lambda=a["reg_lambda"], random_state=seed,
                  n_jobs=4, tree_method="hist", verbosity=0)
    if problem_type == "classification":
        return XGBClassifier(scale_pos_weight=a["scale_pos_weight"] or 1.0, eval_metric="logloss", **common)
    return XGBRegressor(**common)


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    kind: Literal["model", "editor"] = "model"
    task_types: frozenset[str] = frozenset({"tabular"})
    problem_types: frozenset[str] = frozenset({"classification", "regression"})
    builder: Callable[[dict, str, int], Any] | None = None
    scale: bool = True
    interpretability: str = "medium"
    meta: dict[str, Any] = field(default_factory=dict)

    def schema(self) -> dict[str, Any]:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": params}}

    def validate_args(self, args: dict[str, Any] | None) -> dict[str, Any]:
        return self.args_model.model_validate(args or {}).model_dump()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"инструмент {tool.name} уже зарегистрирован")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"инструмент {name} не найден")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def list(self, *, kind: str | None = None, task_type: str | None = None,
             problem_type: str | None = None) -> list[Tool]:
        out = []
        for t in self._tools.values():
            if kind and t.kind != kind:
                continue
            if task_type and task_type not in t.task_types:
                continue
            if problem_type and t.kind == "model" and task_type != "timeseries" and problem_type not in t.problem_types:
                continue
            out.append(t)
        return out

    def names(self, **kw) -> list[str]:
        return [t.name for t in self.list(**kw)]


def default_registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(Tool("fit_logistic_regression",
                    "Логистическая регрессия: интерпретируемая линейная модель вероятности риска. "
                    "Хороша для малых выборок и когда важна объяснимость коэффициентов.",
                    LogisticRegressionArgs, problem_types=frozenset({"classification"}), builder=_build_logreg,
                    interpretability="high"))
    r.register(Tool("fit_ridge_regression",
                    "Ridge-регрессия итоговой оценки: линейная, устойчива к мультиколлинеарности.",
                    RidgeArgs, problem_types=frozenset({"regression"}), builder=_build_ridge, interpretability="high"))
    r.register(Tool("fit_random_forest",
                    "Случайный лес: нелинейные зависимости и взаимодействия, устойчив к выбросам, "
                    "не требует масштабирования.", RandomForestArgs, builder=_build_rf, scale=False))
    r.register(Tool("fit_tabular_xgboost",
                    "Градиентный бустинг XGBoost: обычно наилучшая точность на табличных данных, "
                    "нужен контроль переобучения (max_depth, learning_rate).", XGBoostArgs, builder=_build_xgb,
                    scale=False, interpretability="low"))
    r.register(Tool("fit_arima", "ARIMA/SARIMA для временного ряда.", ArimaArgs, task_types=frozenset({"timeseries"}),
                    problem_types=frozenset({"forecast"})))
    r.register(Tool("fit_ets", "Экспоненциальное сглаживание (Holt-Winters).", EtsArgs,
                    task_types=frozenset({"timeseries"}), problem_types=frozenset({"forecast"})))
    # инструменты Editor-а
    ops_doc = "; ".join(f"{k} — {v}" for k, v in AVAILABLE_OPS.items())
    r.register(Tool("add_features", f"Добавить сгенерированные признаки. Операции: {ops_doc}", AddFeaturesArgs,
                    kind="editor"))
    r.register(Tool("drop_features", "Удалить признаки (например, с утечкой целевой переменной).",
                    DropFeaturesArgs, kind="editor"))
    r.register(Tool("change_threshold", "Изменить порог классификации «в группе риска».", ChangeThresholdArgs,
                    kind="editor"))
    r.register(Tool("change_hyperparams", "Изменить гиперпараметры текущей модели.", ChangeHyperparamsArgs,
                    kind="editor"))
    r.register(Tool("switch_model", "Переключиться на другую модель из реестра.", SwitchModelArgs, kind="editor"))
    return r


DEFAULT_MODEL = {"classification": "fit_logistic_regression", "regression": "fit_ridge_regression",
                 "forecast": "fit_arima"}
