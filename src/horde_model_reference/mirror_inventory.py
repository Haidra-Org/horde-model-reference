"""Small, fail-closed cache for the hash inventory published by the R2 sync."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

import requests
from loguru import logger

__all__ = ["clear_mirror_inventory_cache", "mirror_contains"]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_PATH = "v1/manifest"
_MANIFEST_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class _CachedInventory:
    """Hold validated hashes and their monotonic expiry."""

    hashes: frozenset[str]
    expires_at: float


_CACHE: dict[str, _CachedInventory] = {}
_CACHE_LOCK = threading.Lock()


def clear_mirror_inventory_cache() -> None:
    """Clear process-local inventory state (primarily useful for tests and explicit refreshes)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _validated_hashes(payload: object) -> frozenset[str]:
    """Extract hash keys from schema version 1, rejecting malformed manifests as a whole."""
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported mirror manifest schema")
    objects = payload.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("mirror manifest objects must be a mapping")
    hashes = frozenset(str(value).lower() for value in objects)
    if any(_SHA256_PATTERN.fullmatch(value) is None for value in hashes):
        raise ValueError("mirror manifest contains an invalid SHA-256 key")
    return hashes


def mirror_contains(gateway_base_url: str, sha256: str, *, ttl_seconds: int = 300) -> bool:
    """Return whether the gateway's last valid inventory contains *sha256*.

    Network, HTTP, and validation failures return ``False`` so the mirror remains an optional accelerator and
    clients use the authoritative origin. The API key is never involved in this public inventory request.
    """
    normalized_gateway = gateway_base_url.rstrip("/")
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(normalized_gateway)
        if cached is not None and cached.expires_at > now:
            return sha256.lower() in cached.hashes

    try:
        response = requests.get(
            f"{normalized_gateway}/{_MANIFEST_PATH}",
            allow_redirects=False,
            timeout=_MANIFEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        hashes = _validated_hashes(response.json())
    except (requests.RequestException, ValueError):
        logger.debug("Mirror inventory unavailable or invalid; using origin-only downloads")
        return False

    expiry = now + max(1, ttl_seconds)
    with _CACHE_LOCK:
        _CACHE[normalized_gateway] = _CachedInventory(hashes=hashes, expires_at=expiry)
    return sha256.lower() in hashes
