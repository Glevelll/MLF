"""Сохранение решений преподавателя и пополнение базы знаний подтверждёнными кейсами."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Feedback, HitlReview, KnowledgeBase
from backend.hitl.active_learning import resolve


async def apply_review_decision(session: AsyncSession, review: HitlReview, *, decision: str,
                                adjusted_risk: float | None, comment: str, recommendations: list[str] | None,
                                user_id: int | None) -> dict[str, Any]:
    """decision: confirm | adjust | reject. confirm/adjust добавляют кейс студента в базу знаний."""
    status = {"confirm": "confirmed", "adjust": "adjusted", "reject": "rejected"}[decision]
    payload = review.payload or {}
    final_risk = adjusted_risk if decision == "adjust" and adjusted_risk is not None else payload.get("score")
    dec = {"decision": decision, "adjusted_risk": adjusted_risk, "final_risk": final_risk, "comment": comment}
    await resolve(session, review, status, dec, user_id)
    session.add(Feedback(review_id=review.id, task_id=review.task_id, user_id=user_id, comment=comment,
                         correction={"decision": decision, "adjusted_risk": adjusted_risk,
                                     "recommendations": recommendations}))
    kb_added = False
    if decision in ("confirm", "adjust") and review.kind == "student_risk":
        source_id = f"hitl_{review.task_id}_{payload.get('student_id')}"
        exists = (await session.execute(select(KnowledgeBase.id).where(KnowledgeBase.source_id == source_id))).first()
        if not exists:
            at_risk = (final_risk or 0) >= 0.5
            recs = recommendations or payload.get("recommendations", [])
            status_txt = "в группе риска (подтверждено преподавателем)" if at_risk else "риск снят преподавателем"
            session.add(KnowledgeBase(
                source_id=source_id, kind="student_case", task_type="tabular", problem_type="classification",
                series_features=payload.get("profile", {}), best_model=None, hyperparams={},
                metrics={"risk": final_risk},
                description=f"{payload.get('description', '')} Решение преподавателя: {status_txt}. {comment}".strip(),
                meta={"source": "hitl", "status": "at_risk" if at_risk else "ok", "outcome": final_risk,
                      "recommendations": recs, "confirmed": True, "task_id": review.task_id},
            ))
            kb_added = True
    return {"status": status, "kb_added": kb_added, **dec}
