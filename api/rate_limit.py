"""
api/rate_limit.py — Per-user rate limiting (free vs paid tier).

Enforces:
  - Requests per day limit
  - Requests per minute limit
  - Max tokens per request limit

Uses in-memory counters (Redis in production, dict for development).
"""

import time
from collections import defaultdict
from datetime import datetime, date
from typing import Dict, Tuple

from fastapi import HTTPException

from api.database import User, UsageLog, TIER_LIMITS


# ── In-memory rate limit store ────────────────────────────────────────────────
# In production, replace this with Redis for multi-process support

class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter.
    Tracks requests per minute and per day per user.
    """

    def __init__(self):
        # {user_id: {"minute": (count, timestamp), "day": (count, date)}}
        self._store: Dict[int, dict] = defaultdict(lambda: {
            "minute_count": 0,
            "minute_ts":    0.0,
            "day_count":    0,
            "day_date":     None,
        })

    def check_and_increment(self, user_id: int, tier: str) -> None:
        """
        Check rate limits for a user. Raises HTTPException if exceeded.
        Increments counters on success.

        Args:
            user_id: User's database ID
            tier:    User's subscription tier ("free", "pro", "team")
        """
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
        state  = self._store[user_id]
        now    = time.time()
        today  = date.today()

        # ── Per-minute check ──────────────────────────────────────
        # Reset counter if more than 60 seconds have passed
        if now - state["minute_ts"] > 60:
            state["minute_count"] = 0
            state["minute_ts"]    = now

        if state["minute_count"] >= limits["requests_per_minute"]:
            reset_in = int(60 - (now - state["minute_ts"]))
            raise HTTPException(
                status_code=429,
                detail={
                    "error":   "rate_limit_exceeded",
                    "message": f"Rate limit: {limits['requests_per_minute']} requests/minute exceeded",
                    "retry_after_seconds": reset_in,
                    "upgrade_url": "/pricing",
                },
                headers={"Retry-After": str(reset_in)},
            )

        # ── Per-day check ─────────────────────────────────────────
        if state["day_date"] != today:
            state["day_count"] = 0
            state["day_date"]  = today

        if state["day_count"] >= limits["requests_per_day"]:
            raise HTTPException(
                status_code=429,
                detail={
                    "error":       "daily_limit_exceeded",
                    "message":     f"Daily limit of {limits['requests_per_day']} requests exceeded",
                    "resets_at":   "midnight UTC",
                    "upgrade_url": "/pricing",
                    "tip": (
                        "Upgrade to Pro for unlimited requests: $9.99/month"
                        if tier == "free"
                        else "Contact support for higher limits"
                    ),
                },
            )

        # ── Increment ─────────────────────────────────────────────
        state["minute_count"] += 1
        state["day_count"]    += 1

    def get_remaining(self, user_id: int, tier: str) -> dict:
        """Return current usage stats for a user."""
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])
        state  = self._store[user_id]
        today  = date.today()

        day_count = state["day_count"] if state["day_date"] == today else 0

        return {
            "requests_today":     day_count,
            "requests_today_max": limits["requests_per_day"],
            "remaining_today":    max(0, limits["requests_per_day"] - day_count),
        }


# ── Token limit checker ────────────────────────────────────────────────────────

def check_token_limit(requested_tokens: int, tier: str) -> int:
    """
    Enforce max tokens per request for the user's tier.
    Returns the allowed max_tokens (clamps if over limit).

    Args:
        requested_tokens: Tokens requested in the API call
        tier:             User's tier

    Returns:
        Effective max_tokens to use
    """
    limit = TIER_LIMITS.get(tier, TIER_LIMITS["free"])["max_tokens"]
    if requested_tokens > limit:
        return limit  # Silently cap (or raise 400 — your choice)
    return requested_tokens


# ── Global rate limiter instance ──────────────────────────────────────────────

rate_limiter = InMemoryRateLimiter()
