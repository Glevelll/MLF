"""HITL: очередь студентов высокого риска и задач с низким качеством (роли editor/admin)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import require_editor
from backend.api.schemas import ReviewDecision, ReviewOut
from backend.db.models import HitlReview, User
from backend.db.session import get_session
from backend.hitl import active_learning
from backend.hitl.feedback import apply_review_decision

router = APIRouter(prefix="/api/v1/hitl", tags=["hitl"])


@router.get("/queue", response_model=list[ReviewOut])
async def queue(kind: str | None = None, task_id: str | None = None, limit: int = 50, offset: int = 0,
                _: User = Depends(require_editor), session: AsyncSession = Depends(get_session)) -> list[HitlReview]:
    return await active_learning.list_pending(session, kind, task_id, limit, offset)


@router.get("/stats")
async def stats(_: User = Depends(require_editor), session: AsyncSession = Depends(get_session)) -> dict:
    return await active_learning.stats(session)


@router.post("/reviews/{review_id}")
async def decide(review_id: int, body: ReviewDecision, user: User = Depends(require_editor),
                 session: AsyncSession = Depends(get_session)) -> dict:
    review = await session.get(HitlReview, review_id)
    if review is None:
        raise HTTPException(404, "запись не найдена")
    if review.status != "pending":
        raise HTTPException(409, "запись уже обработана")
    if review.kind != "student_risk":
        raise HTTPException(400, "решения по задачам принимаются через /api/v1/forecast/{task_id}/hitl")
    res = await apply_review_decision(session, review, decision=body.decision, adjusted_risk=body.adjusted_risk,
                                      comment=body.comment, recommendations=body.recommendations, user_id=user.id)
    await session.commit()
    return {"review_id": review_id, **res}
