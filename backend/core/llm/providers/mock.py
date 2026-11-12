"""Детерминированный мок-провайдер.

Имитирует «разумную» LLM без сети: читает роль агента из системного промпта
([ROLE:...]) и JSON-контекст из блока <context>...</context>, применяет эвристическую
политику и возвращает tool calls / JSON как настоящая модель. Позволяет запускать
весь пайплайн и тесты без Ollama. Параметр fail_times имитирует невалидные ответы
слабой модели (для проверки fallback-логики).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from backend.core.llm.base import LLMProvider, LLMResponse, Message, ToolCall

_ROLE = re.compile(r"\[ROLE:([a-z_]+)\]")
_CTX = re.compile(r"<context>(.*?)</context>", re.S)

INTERPRETABILITY = {
    "fit_logistic_regression": 9.0,
    "fit_ridge_regression": 9.0,
    "fit_random_forest": 6.5,
    "fit_tabular_xgboost": 5.5,
    "fit_arima": 7.5,
    "fit_ets": 8.0,
}


def _parse(messages: list[Message]) -> tuple[str, dict[str, Any]]:
    role, ctx = "generic", {}
    for m in messages:
        if m.role == "system" and (r := _ROLE.search(m.content)):
            role = r.group(1)
    for m in reversed(messages):
        if m.role == "user" and (c := _CTX.search(m.content)):
            try:
                ctx = json.loads(c.group(1))
            except json.JSONDecodeError:
                ctx = {}
            break
    return role, ctx


def select_model_policy(ctx: dict[str, Any]) -> ToolCall:
    tools: list[str] = ctx.get("available_tools", [])
    meta = ctx.get("meta", {})
    votes: Counter[str] = Counter()
    params: dict[str, dict] = {}
    for rank, case in enumerate(ctx.get("rag_cases", [])):
        tool = case.get("best_model")
        if tool in tools:
            votes[tool] += 1.0 / (rank + 1)
            params.setdefault(tool, case.get("hyperparams") or {})
    if votes:
        name = votes.most_common(1)[0][0]
        return ToolCall(name=name, arguments=params.get(name, {}))
    n_rows = meta.get("n_rows") or 0
    prefs = (["fit_logistic_regression", "fit_ridge_regression"] if n_rows < 400
             else ["fit_tabular_xgboost", "fit_random_forest"])
    for p in prefs + tools:
        if p in tools:
            args = {"class_weight": "balanced"} if p in ("fit_logistic_regression",) else {}
            return ToolCall(name=p, arguments=args)
    return ToolCall(name=tools[0] if tools else "fit_logistic_regression", arguments={})


def critic_policy(ctx: dict[str, Any]) -> dict[str, Any]:
    problem = ctx.get("problem_type", "classification")
    m = ctx.get("metrics", {})
    if problem == "classification":
        v = m.get("roc_auc") or 0.5
        quality = max(0.0, min(10.0, (v - 0.5) / 0.45 * 10))
    else:
        v = m.get("r2") or 0.0
        quality = max(0.0, min(10.0, v * 11))
    issues = ctx.get("issues", [])
    crit = sum(1 for i in issues if i.get("severity") == "critical")
    warn = sum(1 for i in issues if i.get("severity") == "warning")
    interp = INTERPRETABILITY.get(ctx.get("model_type", ""), 6.0)
    if len(ctx.get("top_features", [])) >= 3:
        interp = min(10.0, interp + 0.5)
    usefulness = max(0.0, quality - 2.5 * crit - 0.5 * warn)
    score = round(0.5 * quality + 0.25 * interp + 0.25 * usefulness - 2.0 * crit, 2)
    score = max(0.0, min(10.0, score))
    top = ", ".join(f["feature"] for f in ctx.get("top_features", [])[:3]) or "нет данных"
    reasoning = (f"Качество модели {quality:.1f}/10 (основная метрика {v:.3f}); интерпретируемость {interp:.1f}/10; "
                 f"ключевые факторы риска: {top}. Критических замечаний: {crit}, предупреждений: {warn}.")
    suggestions = []
    if crit:
        suggestions.append("Устранить критические замечания валидатора (утечки/калибровка) перед использованием.")
    if quality < 6:
        suggestions.append("Добавить признаки динамики оценок и активности; попробовать ансамбль деревьев.")
    if interp < 7:
        suggestions.append("Сопровождать прогноз объяснением локальных факторов риска для преподавателя.")
    return {"score": score, "quality": round(quality, 2), "interpretability": interp,
            "decision_usefulness": round(usefulness, 2), "critical": crit > 0 or score < 3,
            "reasoning": reasoning, "suggestions": suggestions}


def editor_policy(ctx: dict[str, Any]) -> list[ToolCall]:
    codes = {i.get("code") for i in ctx.get("issues", [])}
    calls: list[ToolCall] = []
    done_ops = set(ctx.get("feature_ops", []))
    avail_ops = ctx.get("available_ops", [])
    if "leakage" in codes and ctx.get("leaking_features"):
        calls.append(ToolCall(name="drop_features", arguments={"columns": ctx["leaking_features"],
                                                                  "reason": "утечка целевой переменной"}))
    new_ops = [o for o in ("score_dynamics", "activity_aggregates", "ratios") if o in avail_ops and o not in done_ops]
    if new_ops and ("low_quality" in codes or ctx.get("critic_score", 10) < ctx.get("critic_threshold", 6)):
        calls.append(ToolCall(name="add_features", arguments={"ops": new_ops[:2]}))
    if ctx.get("problem_type") == "classification" and ctx.get("best_f1_threshold") is not None:
        if abs(ctx["best_f1_threshold"] - ctx.get("decision_threshold", 0.5)) >= 0.05:
            calls.append(ToolCall(name="change_threshold", arguments={"threshold": ctx["best_f1_threshold"]}))
    if "poor_calibration" in codes and ctx.get("model_type") == "fit_tabular_xgboost":
        calls.append(ToolCall(name="change_hyperparams", arguments={"params": {"max_depth": 3, "n_estimators": 200}}))
    if "low_quality" in codes and ctx.get("model_type") in ("fit_logistic_regression", "fit_ridge_regression") \
            and ctx.get("iteration", 0) >= 1:
        calls.append(ToolCall(name="switch_model", arguments={"model_type": "fit_random_forest"}))
    return calls


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, model: str = "mock-policy-v1", fail_times: int = 0) -> None:
        super().__init__(model)
        self.fail_times = fail_times
        self.calls = 0

    async def chat(self, messages, *, tools=None, json_mode=False, temperature=0.0) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.fail_times:
            return LLMResponse(content="Извините, я думаю что лучше подойдёт какая-то модель.",
                               model=self.model, provider=self.name)
        role, ctx = _parse(messages)
        if role == "model_selector":
            call = select_model_policy(ctx)
            return LLMResponse(tool_calls=[call], model=self.model, provider=self.name)
        if role == "critic":
            return LLMResponse(content=json.dumps(critic_policy(ctx), ensure_ascii=False),
                               model=self.model, provider=self.name)
        if role == "editor":
            calls = editor_policy(ctx)
            if not calls:
                return LLMResponse(content='{"tool_calls": []}', model=self.model, provider=self.name)
            return LLMResponse(tool_calls=calls, model=self.model, provider=self.name)
        if role == "explainer":
            drivers = ", ".join(ctx.get("drivers", [])[:3]) or "совокупность факторов"
            return LLMResponse(content=f"Риск обусловлен: {drivers}. Рекомендуется индивидуальная консультация.",
                               model=self.model, provider=self.name)
        return LLMResponse(content="OK", model=self.model, provider=self.name)
