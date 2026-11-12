"""Очередь неоднозначных/рискованных примеров для преподавателя (active learning)."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import HitlReview


async def enqueue_students(session: AsyncSession, task_id: str, students: list[dict[str, Any]]) -> int:
    for s in students:
        session.add(HitlReview(task_id=task_id, kind="student_risk", priority=float(s.get("score", 0)), payload=s))
    return len(students)


async def enqueue_task_quality(session: AsyncSession, task_id: str, reason: str, summary: dict[str, Any]) -> HitlReview:
    row = HitlReview(task_id=task_id, kind="task_quality", priority=1.0, payload={"reason": reason, **summary})
    session.add(row)
    return row


async def list_pending(session: AsyncSession, kind: str | None = None, task_id: str | None = None,
                       limit: int = 50, offset: int = 0) -> list[HitlReview]:
    q = select(HitlReview).where(HitlReview.status == "pending")
    if kind:
        q = q.where(HitlReview.kind == kind)
    if task_id:
        q = q.where(HitlReview.task_id == task_id)
    q = q.order_by(HitlReview.priority.desc(), HitlReview.id).limit(limit).offset(offset)
    return list((await session.execute(q)).scalars().all())


async def stats(session: AsyncSession) -> dict[str, int]:
    rows = (await session.execute(select(HitlReview.status, func.count()).group_by(HitlReview.status))).all()
    return {status: int(n) for status, n in rows}


async def resolve(session: AsyncSession, review: HitlReview, status: str, decision: dict[str, Any],
                  reviewer_id: int | None) -> HitlReview:
    review.status = status
    review.decision = decision
    review.reviewer_id = reviewer_id
    review.resolved_at = datetime.now(UTC)
    return review
