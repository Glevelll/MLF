"""Создаёт демо-пользователя для локальной разработки (первый пользователь получает роль admin).

python scripts/seed_demo_user.py [--username demo --password demo12345]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from backend.db.models import User  # noqa: E402
from backend.db.session import dispose_engine, init_db, session_scope  # noqa: E402
from backend.utils.security import hash_password  # noqa: E402

DEMO_USERNAME = "demo"
DEMO_PASSWORD = "demo12345"  # только для локальной разработки


async def seed(username: str, password: str, role: str) -> str:
    await init_db()
    async with session_scope() as s:
        user = (await s.execute(select(User).where(User.username == username))).scalar_one_or_none()
        if user is None:
            s.add(User(email=f"{username}@example.com", username=username, hashed_password=hash_password(password),
                       role=role))
            msg = f"создан пользователь {username} ({role})"
        else:
            user.role = role
            msg = f"пользователь {username} уже существует, роль {role}"
    await dispose_engine()
    return msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default=DEMO_USERNAME)
    ap.add_argument("--password", default=DEMO_PASSWORD)
    ap.add_argument("--role", default="admin", choices=["user", "editor", "admin"])
    a = ap.parse_args()
    print(asyncio.run(seed(a.username, a.password, a.role)))
