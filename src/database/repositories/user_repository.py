"""SQLAlchemy repository for API therapist and administrator accounts."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.database.models import ApiUser


def get_user(session: Session, user_id: int):
    return session.get(ApiUser, user_id)


def get_user_by_username(session: Session, username: str):
    return session.scalar(
        select(ApiUser).where(func.lower(ApiUser.username) == username.strip().lower())
    )


def create_user(session: Session, *, username: str, password_hash: str, role: str):
    user = ApiUser(
        username=username.strip().lower(),
        password_hash=password_hash,
        role=role,
        is_active=1,
    )
    session.add(user)
    session.flush()
    return user
