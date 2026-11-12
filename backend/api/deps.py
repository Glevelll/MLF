"""Зависимости FastAPI: сессия БД, текущий пользователь, проверка ролей."""
from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import User
from backend.db.session import get_session
from backend.utils.security import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme),
                           session: AsyncSession = Depends(get_session)) -> User:
    cred_exc = HTTPException(status.HTTP_401_UNAUTHORIZED, "неверный или просроченный токен",
                             headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = decode_token(token)
    except jwt.PyJWTError as e:
        raise cred_exc from e
    user = (await session.execute(select(User).where(User.username == payload.get("sub")))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise cred_exc
    return user


def require_roles(*roles: str):
    async def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"требуется роль: {', '.join(roles)}")
        return user

    return checker


require_admin = require_roles("admin")
require_editor = require_roles("editor", "admin")
