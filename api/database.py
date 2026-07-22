"""
api/database.py — Database models and session management.

Uses SQLite for development, easily upgradable to PostgreSQL for production.
Stores: users, API keys, usage logs.
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer,
    DateTime, Boolean, Float, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./se_llm_api.db")
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Database Models ───────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id           = Column(Integer, primary_key=True, index=True)
    email        = Column(String, unique=True, index=True, nullable=False)
    name         = Column(String, nullable=True)
    tier         = Column(String, default="free")       # "free" | "pro" | "team"
    created_at   = Column(DateTime, default=datetime.utcnow)
    is_active    = Column(Boolean, default=True)
    stripe_id    = Column(String, nullable=True)        # Stripe customer ID


class APIKey(Base):
    __tablename__ = "api_keys"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, nullable=False, index=True)
    key_hash     = Column(String, unique=True, nullable=False)  # hashed key
    key_prefix   = Column(String, nullable=False)               # first 8 chars (display)
    name         = Column(String, default="default")            # key label
    created_at   = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    is_active    = Column(Boolean, default=True)


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, nullable=False, index=True)
    api_key_id      = Column(Integer, nullable=False)
    timestamp       = Column(DateTime, default=datetime.utcnow, index=True)
    endpoint        = Column(String, nullable=False)
    prompt_tokens   = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens    = Column(Integer, default=0)
    latency_ms      = Column(Float, default=0)
    model           = Column(String, default="se-llm-350m")


# ── Tier limits ───────────────────────────────────────────────────────────────

TIER_LIMITS = {
    "free": {
        "requests_per_day":   50,
        "max_tokens":         512,
        "requests_per_minute": 5,
    },
    "pro": {
        "requests_per_day":   10_000,
        "max_tokens":         2048,
        "requests_per_minute": 60,
    },
    "team": {
        "requests_per_day":   50_000,
        "max_tokens":         2048,
        "requests_per_minute": 200,
    },
}


# ── Session dependency ────────────────────────────────────────────────────────

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    print("Database initialized")


if __name__ == "__main__":
    init_db()
    print("Tables created successfully")
