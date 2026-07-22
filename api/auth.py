"""
api/auth.py — API Key Authentication and User Management.

Handles:
  - API key generation (cryptographically random)
  - Key hashing (SHA-256, keys never stored in plaintext)
  - Key validation on every request
  - User registration
"""

import os
import secrets
import hashlib
from datetime import datetime
from typing import Optional

from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from api.database import APIKey, User, get_db

# ── API key header ────────────────────────────────────────────────────────────

API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)
KEY_PREFIX = "sellm-"   # Prefix for all API keys (easy to identify)


# ── Key generation ────────────────────────────────────────────────────────────

def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key.

    Returns:
        (raw_key, key_hash)
        raw_key:  shown to user ONCE, never stored
        key_hash: stored in DB for verification
    """
    raw_key  = KEY_PREFIX + secrets.token_urlsafe(32)
    key_hash = _hash_key(raw_key)
    return raw_key, key_hash


def _hash_key(raw_key: str) -> str:
    """SHA-256 hash of an API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


# ── User registration ─────────────────────────────────────────────────────────

def register_user(email: str, name: str, db: Session) -> tuple[User, str]:
    """
    Register a new user and generate their first API key.

    Returns:
        (user, raw_api_key)
        raw_api_key is shown once — user must save it
    """
    # Check if user already exists
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user (free tier by default)
    user = User(email=email, name=name, tier="free")
    db.add(user)
    db.flush()  # Get user.id without committing

    # Generate API key
    raw_key, key_hash = generate_api_key()
    api_key = APIKey(
        user_id    = user.id,
        key_hash   = key_hash,
        key_prefix = raw_key[:12],  # Store first 12 chars for display
        name       = "default",
    )
    db.add(api_key)
    db.commit()
    db.refresh(user)

    return user, raw_key


# ── Key validation ────────────────────────────────────────────────────────────

def validate_api_key(
    authorization: Optional[str] = Security(API_KEY_HEADER),
    db: Session = Depends(get_db),
) -> tuple[User, APIKey]:
    """
    FastAPI dependency: validate API key from Authorization header.

    Expected header format:
        Authorization: Bearer sellm-xxxxxxxxxxxx

    Returns:
        (user, api_key) if valid
    Raises:
        HTTPException 401 if invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include: Authorization: Bearer <your-key>",
        )

    # Strip "Bearer " prefix if present
    raw_key = authorization
    if raw_key.lower().startswith("bearer "):
        raw_key = raw_key[7:]

    raw_key = raw_key.strip()

    if not raw_key.startswith(KEY_PREFIX):
        raise HTTPException(status_code=401, detail="Invalid API key format")

    # Hash and look up in DB
    key_hash = _hash_key(raw_key)
    api_key  = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == True,
    ).first()

    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    # Look up user
    user = db.query(User).filter(User.id == api_key.user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found or deactivated")

    # Update last used timestamp
    api_key.last_used_at = datetime.utcnow()
    db.commit()

    return user, api_key
