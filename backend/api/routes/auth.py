"""Регистрация, вход (JWT), профиль, управление ролями."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, require_admin
from backend.api.schemas import RegisterRequest, TokenOut, UserOut
from backend.core.config import get_settings
from backend.db.models import User
from backend.db.session import get_session
from backend.utils.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, session: AsyncSession = Depends(get_session)) -> User:
    exists = (await session.execute(select(User).where(or_(User.email == req.email,
                                                            User.username == req.username)))).first()
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "пользователь с таким email или именем уже существует")
    n_users = (await session.execute(select(func.count(User.id)))).scalar_one()
    admin_email = get_settings().admin_email
    role = "admin" if n_users == 0 or (admin_email and req.email == admin_email) else "user"
    user = User(email=req.email, username=req.username, hashed_password=hash_password(req.password), role=role)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/login", response_model=TokenOut)
async def login(form: OAuth2PasswordRequestForm = Depends(), session: AsyncSession = Depends(get_session)) -> TokenOut:
    user = (await session.execute(select(User).where(or_(User.username == form.username,
                                                          User.email == form.username)))).scalar_one_or_none()
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "неверный логин или пароль")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "пользователь заблокирован")
    return TokenOut(access_token=create_access_token(user.username, user.role), role=user.role)


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)) -> User:
    return user


@router.post("/users/{user_id}/role", response_model=UserOut)
async def set_role(user_id: int, role: str, _: User = Depends(require_admin),
                   session: AsyncSession = Depends(get_session)) -> User:
    if role not in ("user", "editor", "admin"):
        raise HTTPException(400, "роль должна быть user | editor | admin")
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(404, "пользователь не найден")
    user.role = role
    await session.commit()
    await session.refresh(user)
    return user
