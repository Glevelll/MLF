"""База знаний: профили студентов, кейсы выбора модели, описания и рекомендации."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

PROFILE_KEYS = ["prior_failures", "absence", "effort", "score_first", "score_last", "score_trend", "hw",
                "support", "aspiration", "alcohol", "health", "social", "late", "internet"]


def _num(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _yes(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return 1.0 if str(v).lower() in ("yes", "y", "1", "true") else 0.0


def canonical_profile(row: dict[str, Any]) -> dict[str, float | None]:
    """Приводит разные схемы (UCI, OULAD-подобная, пользовательская) к общему профилю 0..1."""
    p: dict[str, float | None] = dict.fromkeys(PROFILE_KEYS)
    p["prior_failures"] = _num(row, "failures") if "failures" in row else _num(row, "num_prev_attempts")
    if (a := _num(row, "absences")) is not None:
        p["absence"] = min(a / 30, 1.0)
    elif (a := _num(row, "attendance_rate")) is not None:
        p["absence"] = 1 - a
    if (s := _num(row, "studytime")) is not None:
        p["effort"] = s / 4
    elif (s := _num(row, "lms_hours")) is not None:
        p["effort"] = min(s / 40, 1.0)
    first = _num(row, "G1")
    last = _num(row, "G2")
    if first is not None:
        first, last = first / 20, (last / 20 if last is not None else None)
    else:
        first = _num(row, "test1_score")
        last = _num(row, "test3_score") or _num(row, "test2_score")
        first = first / 100 if first is not None else None
        last = last / 100 if last is not None else None
    p["score_first"], p["score_last"] = first, last
    if first is not None and last is not None:
        p["score_trend"] = last - first
    if (h := _num(row, "hw_avg_score")) is not None:
        p["hw"] = h / 100
    elif (h := _num(row, "hw_submitted_ratio")) is not None:
        p["hw"] = h
    sup = [x for x in (_yes(row, "schoolsup"), _yes(row, "famsup"), _yes(row, "scholarship")) if x is not None]
    p["support"] = max(sup) if sup else None
    p["aspiration"] = _yes(row, "higher")
    dalc, walc = _num(row, "Dalc"), _num(row, "Walc")
    if dalc is not None and walc is not None:
        p["alcohol"] = (dalc + walc) / 10
    if (h := _num(row, "health")) is not None:
        p["health"] = h / 5
    if (g := _num(row, "goout")) is not None:
        p["social"] = g / 5
    elif (f := _num(row, "forum_posts")) is not None:
        p["social"] = min(f / 10, 1.0)
    if (lt := _num(row, "late_submissions")) is not None:
        p["late"] = min(lt / 5, 1.0)
    p["internet"] = _yes(row, "internet")
    return p


def profile_vector(profile: dict[str, float | None]) -> np.ndarray:
    return np.array([np.nan if profile.get(k) is None else float(profile[k]) for k in PROFILE_KEYS])


def _level(x: float | None, lo: float, hi: float, words=("низкий", "средний", "высокий")) -> str | None:
    if x is None:
        return None
    return words[0] if x < lo else (words[2] if x > hi else words[1])


def describe_student(profile: dict[str, float | None], raw: dict[str, Any] | None = None) -> str:
    parts = []
    if (f := profile.get("prior_failures")) is not None:
        parts.append("без академических задолженностей" if f == 0 else f"академические задолженности в прошлом: {int(f)}")
    if (a := _level(profile.get("absence"), 0.15, 0.35)) is not None:
        parts.append(f"уровень пропусков {a}")
    if (e := _level(profile.get("effort"), 0.35, 0.65)) is not None:
        parts.append(f"учебная нагрузка/время {e}")
    if profile.get("score_first") is not None:
        parts.append(f"промежуточные оценки {_level(profile['score_first'], 0.5, 0.75)}")
    if (t := profile.get("score_trend")) is not None:
        parts.append("оценки снижаются" if t < -0.05 else ("оценки растут" if t > 0.05 else "оценки стабильны"))
    if (h := _level(profile.get("hw"), 0.55, 0.8)) is not None:
        parts.append(f"выполнение домашних заданий {h}")
    if profile.get("support") == 1:
        parts.append("получает поддержку (семья/школа/стипендия)")
    if profile.get("aspiration") == 0:
        parts.append("не планирует высшее образование")
    if (al := profile.get("alcohol")) is not None and al > 0.5:
        parts.append("повышенное употребление алкоголя")
    if (s := _level(profile.get("social"), 0.3, 0.7)) is not None:
        parts.append(f"социальная активность {s}")
    if (lt := profile.get("late")) is not None and lt > 0.4:
        parts.append("частые поздние сдачи")
    if profile.get("internet") == 0:
        parts.append("нет доступа к интернету дома")
    if raw:
        extra = [f"{k}={raw[k]}" for k in ("sex", "gender", "age", "age_band", "subject", "specialty") if k in raw]
        if extra:
            parts.append("демография: " + ", ".join(map(str, extra)))
    return "Студент: " + "; ".join(parts) + "." if parts else "Студент: профиль без деталей."


def recommendations_for(profile: dict[str, float | None], at_risk: bool) -> list[str]:
    rec = []
    if (a := profile.get("absence")) is not None and a > 0.25:
        rec.append("Контроль посещаемости: беседа куратора, выяснение причин пропусков.")
    if (f := profile.get("prior_failures")) and f > 0:
        rec.append("Индивидуальный план ликвидации задолженностей и тьютор по проблемным темам.")
    if (t := profile.get("score_trend")) is not None and t < -0.05:
        rec.append("Раннее вмешательство: оценки снижаются — консультация преподавателя до итогового контроля.")
    if (e := profile.get("effort")) is not None and e < 0.35:
        rec.append("Увеличить регулярное учебное время: план самостоятельной работы, напоминания в LMS.")
    if (h := profile.get("hw")) is not None and h < 0.6:
        rec.append("Поддержка с домашними заданиями: дополнительные дедлайны, разбор типовых ошибок.")
    if (lt := profile.get("late")) is not None and lt > 0.4:
        rec.append("Тайм-менеджмент: дробные контрольные точки, отслеживание поздних сдач.")
    if (al := profile.get("alcohol")) is not None and al > 0.5:
        rec.append("Направить к психологу/социальной службе вуза.")
    if profile.get("support") == 0 and at_risk:
        rec.append("Рассмотреть академическую поддержку (наставник, стипендия, консультации).")
    if not rec:
        rec.append("Поддерживать текущую траекторию; плановый мониторинг успеваемости." if not at_risk else
                   "Индивидуальная консультация для выявления скрытых причин риска.")
    return rec


def describe_dataset_case(meta: dict[str, Any], best_model: str, metrics: dict[str, Any], source: str) -> str:
    ptype = "классификация (риск неуспеваемости/отчисления)" if meta.get("is_classification") else "регрессия итоговой оценки"
    bits = [f"Набор «{source}»: {ptype}", f"{int(meta.get('n_rows', 0))} строк, {int(meta.get('n_features', 0))} признаков"
            f" ({int(meta.get('n_categorical', 0))} категориальных)"]
    if meta.get("missing_rate", 0) > 0.01:
        bits.append(f"пропуски {meta['missing_rate'] * 100:.1f}%")
    if meta.get("is_classification") and meta.get("minority_ratio", 0.5) < 0.25:
        bits.append(f"дисбаланс классов (миноритарный {meta['minority_ratio'] * 100:.0f}%)")
    if meta.get("max_abs_corr", 0) > 0.7:
        bits.append("есть сильный линейный предиктор (промежуточные оценки)")
    elif meta.get("mean_abs_corr", 0) < 0.1:
        bits.append("слабые линейные связи, вероятны нелинейности")
    if meta.get("has_score_columns"):
        bits.append("есть последовательность промежуточных оценок")
    m = metrics.get("roc_auc") if meta.get("is_classification") else metrics.get("r2")
    name = {"fit_logistic_regression": "логистическая регрессия", "fit_ridge_regression": "Ridge-регрессия",
            "fit_random_forest": "случайный лес", "fit_tabular_xgboost": "XGBoost"}.get(best_model, best_model)
    metric_name = "ROC-AUC" if meta.get("is_classification") else "R²"
    return "; ".join(bits) + f". Лучшая модель: {name} ({metric_name}={m:.3f})." if m is not None else "; ".join(bits)


def contextualize(kind: str, text: str, payload: dict[str, Any]) -> str:
    """Contextual Retrieval: краткий контекст документа добавляется к чанку перед индексацией."""
    if kind == "student_case":
        status = payload.get("status", "")
        ctx = f"[Кейс студента из источника {payload.get('source', '?')}; итог: {status}]"
    elif kind == "dataset_case":
        ctx = (f"[Кейс выбора модели; задача {payload.get('problem_type')}; "
               f"лучшая модель {payload.get('best_model')}]")
    else:
        ctx = f"[{kind}]"
    return f"{ctx} {text}"
