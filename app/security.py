from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

from app.config import (
    JWT_ALGORITHM,
    JWT_EXPIRE_MINUTES,
    JWT_SECRET,
)


password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """
    Convert a plaintext password into a secure Argon2 hash.
    """
    return password_hash.hash(password)


def verify_password(
    password: str,
    hashed_password: str,
) -> bool:
    """
    Verify plaintext password against stored password hash.
    """
    return password_hash.verify(
        password,
        hashed_password,
    )


def create_access_token(
    uid: str,
    email: str,
) -> str:
    """
    Create a signed JWT access token.
    """

    expires_at = (
        datetime.now(timezone.utc)
        + timedelta(minutes=JWT_EXPIRE_MINUTES)
    )

    payload = {
        "sub": uid,
        "email": email,
        "exp": expires_at,
    }

    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict:
    """
    Verify JWT signature and expiration.
    """

    return jwt.decode(
        token,
        JWT_SECRET,
        algorithms=[JWT_ALGORITHM],
    )