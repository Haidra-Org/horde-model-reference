"""Acquire and cache short-lived, mirror-only gateway credentials."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests
from loguru import logger

__all__ = ["clear_gateway_session_cache", "gateway_auth_headers"]

_SESSION_PATH = "v1/session"
_REQUEST_TIMEOUT_SECONDS = 10
_EXPIRY_SKEW_SECONDS = 15


@dataclass(frozen=True)
class _Session:
    """Hold a mirror-only bearer token until shortly before expiry."""

    token: str | None
    expires_at: float


_CACHE: dict[tuple[str, str], _Session] = {}
_LOCK = threading.Lock()


def clear_gateway_session_cache() -> None:
    """Clear cached positive and rollout-fallback session results."""
    with _LOCK:
        _CACHE.clear()


def gateway_auth_headers(gateway_base_url: str, api_key: str) -> dict[str, str]:
    """Return a short-lived bearer header, falling back to the legacy key header during gateway rollout.

    The session exchange and later object request both disable redirects. A gateway without the session endpoint
    remains compatible, but its negative result is cached briefly rather than retried for every file.
    """
    normalized_gateway = gateway_base_url.rstrip("/")
    identity = hashlib.sha256(api_key.encode()).hexdigest()
    cache_key = (normalized_gateway, identity)
    now = time.time()
    with _LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None and cached.expires_at > now:
            return {"Authorization": f"Bearer {cached.token}"} if cached.token else {"apikey": api_key}

    try:
        response = requests.post(
            f"{normalized_gateway}/{_SESSION_PATH}",
            headers={"apikey": api_key, "accept": "application/json"},
            allow_redirects=False,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload: Any = response.json()
        token = payload.get("token") if isinstance(payload, dict) else None
        expires_at = payload.get("expires_at") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token.startswith("hmr1.") or not isinstance(expires_at, int):
            raise ValueError("invalid mirror session response")
        session = _Session(token=token, expires_at=max(now + 1, expires_at - _EXPIRY_SKEW_SECONDS))
    except (requests.RequestException, ValueError):
        logger.debug("Mirror session exchange unavailable; using legacy no-redirect gateway authentication")
        session = _Session(token=None, expires_at=now + 60)

    with _LOCK:
        _CACHE[cache_key] = session
    return {"Authorization": f"Bearer {session.token}"} if session.token else {"apikey": api_key}
