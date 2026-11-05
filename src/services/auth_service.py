"""Database-backed API authentication using PBKDF2 and signed JWTs."""

import hashlib
import hmac
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import jwt

from src.database import db

ALGORITHM = "HS256"
PASSWORD_ITERATIONS = 600_000
TOKEN_MINUTES = 30


class AuthenticationError(ValueError):
    pass


class UserConflictError(ValueError):
    pass


def ensure_auth_schema():
    conn = db.get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('therapist', 'admin')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()


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
    return "pbkdf2_sha256$" + str(PASSWORD_ITERATIONS) + "$" + salt.hex() + "$" + digest.hex()


def _verify_password(password, encoded):
    try:
        scheme, iterations, salt_hex, digest_hex = encoded.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(actual.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def create_user(username, password, role):
    ensure_auth_schema()
    conn = db.get_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO api_users(username, password_hash, role)
               VALUES (?, ?, ?)""",
            (username.strip().lower(), _hash_password(password), role),
        )
        conn.commit()
        return get_user(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise UserConflictError("That username already exists.") from exc
    finally:
        conn.close()


def get_user(user_id):
    conn = db.get_connection()
    try:
        row = conn.execute(
            """SELECT user_id, username, role, is_active
               FROM api_users WHERE user_id = ?""",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "user_id": row[0], "username": row[1],
            "role": row[2], "is_active": bool(row[3]),
        }
    finally:
        conn.close()


def authenticate(username, password):
    ensure_auth_schema()
    conn = db.get_connection()
    try:
        row = conn.execute(
            """SELECT user_id, username, password_hash, role, is_active
               FROM api_users WHERE username = ? COLLATE NOCASE""",
            (username.strip(),),
        ).fetchone()
    finally:
        conn.close()
    if row is None or not row[4] or not _verify_password(password, row[2]):
        raise AuthenticationError("Invalid username or password.")
    return {"user_id": row[0], "username": row[1], "role": row[3]}


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


def decode_token(token):
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, TypeError, ValueError) as exc:
        raise AuthenticationError("Invalid or expired access token.") from exc
    user = get_user(user_id)
    if user is None or not user["is_active"]:
        raise AuthenticationError("User is inactive or no longer exists.")
    return user


def bootstrap_admin_from_environment():
    username = os.environ.get("NARS_BOOTSTRAP_ADMIN_USERNAME", "").strip()
    password = os.environ.get("NARS_BOOTSTRAP_ADMIN_PASSWORD", "")
    if not username and not password:
        return
    if not username or len(password) < 12:
        raise RuntimeError(
            "Set both NARS_BOOTSTRAP_ADMIN_USERNAME and a 12+ character "
            "NARS_BOOTSTRAP_ADMIN_PASSWORD, or leave both unset."
        )
    ensure_auth_schema()
    conn = db.get_connection()
    try:
        exists = conn.execute(
            "SELECT 1 FROM api_users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()
    finally:
        conn.close()
    if not exists:
        create_user(username, password, "admin")
