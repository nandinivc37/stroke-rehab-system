"""SQLAlchemy-backed API authentication using PBKDF2 and signed JWTs."""

import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database.models import ApiUser
from src.database.repositories import user_repository

ALGORITHM = "HS256"
PASSWORD_ITERATIONS = 600_000
TOKEN_MINUTES = 30


class AuthenticationError(ValueError):
    pass


class UserConflictError(ValueError):
    pass


def ensure_auth_schema(engine):
    """Ensure the API account table exists for the local SQLite transition.

    Production databases should be created and upgraded with Alembic. The
    check-first create is retained for the existing local SQLite workflow.
    """
    ApiUser.__table__.create(engine, checkfirst=True)


def _secret():
    value = os.environ.get("NARS_JWT_SECRET", "")
    if len(value) < 32:
        raise RuntimeError(
            "NARS_JWT_SECRET must be configured with at least 32 characters."
        )
    return value


def _hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return (
        "pbkdf2_sha256$" + str(PASSWORD_ITERATIONS) + "$"
        + salt.hex() + "$" + digest.hex()
    )


def _verify_password(password, encoded):
    try:
        scheme, iterations, salt_hex, digest_hex = encoded.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(actual.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def _user_read(user: ApiUser):
    return {
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role,
        "is_active": bool(user.is_active),
    }


def create_user(session: Session, username, password, role):
    try:
        user = user_repository.create_user(
            session,
            username=username,
            password_hash=_hash_password(password),
            role=role,
        )
        session.commit()
        session.refresh(user)
        return _user_read(user)
    except IntegrityError as exc:
        session.rollback()
        raise UserConflictError("That username already exists.") from exc


def get_user(session: Session, user_id):
    user = user_repository.get_user(session, user_id)
    return _user_read(user) if user is not None else None


def authenticate(session: Session, username, password):
    user = user_repository.get_user_by_username(session, username)
    if (
        user is None
        or not user.is_active
        or not _verify_password(password, user.password_hash)
    ):
        raise AuthenticationError("Invalid username or password.")
    return _user_read(user)


def issue_token(user):
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=TOKEN_MINUTES)
    token = jwt.encode(
        {
            "sub": str(user["user_id"]),
            "username": user["username"],
            "role": user["role"],
            "iat": now,
            "exp": expires,
        },
        _secret(),
        algorithm=ALGORITHM,
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": TOKEN_MINUTES * 60,
    }


def decode_token(session: Session, token):
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, TypeError, ValueError) as exc:
        raise AuthenticationError("Invalid or expired access token.") from exc

    # Resolve the account on every request so role changes and deactivation
    # take effect immediately; never trust role claims embedded in the JWT.
    user = get_user(session, user_id)
    if user is None or not user["is_active"]:
        raise AuthenticationError("User is inactive or no longer exists.")
    return user


def bootstrap_admin_from_environment(session_factory):
    username = os.environ.get("NARS_BOOTSTRAP_ADMIN_USERNAME", "").strip()
    password = os.environ.get("NARS_BOOTSTRAP_ADMIN_PASSWORD", "")
    if not username and not password:
        return
    if not username or len(password) < 12:
        raise RuntimeError(
            "Set both NARS_BOOTSTRAP_ADMIN_USERNAME and a 12+ character "
            "NARS_BOOTSTRAP_ADMIN_PASSWORD, or leave both unset."
        )

    session = session_factory()
    try:
        existing = user_repository.get_user_by_username(session, username)
        if existing is None:
            create_user(session, username, password, "admin")
    finally:
        session.close()
