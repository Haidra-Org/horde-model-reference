"""Backend providers for model reference data sources.

This module defines the abstract backend interface and concrete implementations
for fetching model reference data from different sources:

- FileSystemBackend: PRIMARY mode - reads/writes local JSON files
- GitHubBackend: REPLICA mode - downloads from GitHub with legacy conversion
- HTTPBackend: REPLICA mode - fetches from PRIMARY API with GitHub fallback
- LocalReadOnlyBackend: REPLICA mode - reads local JSON files, never downloads (offline subprocesses)
- RedisBackend: PRIMARY mode wrapper - adds distributed caching (requires `redis` extra)

Each backend is designed for specific replication modes and use cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ModelReferenceBackend
from .filesystem_backend import FileSystemBackend
from .github_backend import GitHubBackend
from .http_backend import HTTPBackend
from .local_readonly_backend import LocalReadOnlyBackend
from .replica_backend_base import ReplicaBackendBase

if TYPE_CHECKING:
    from .redis_backend import RedisBackend

__all__ = [
    "FileSystemBackend",
    "GitHubBackend",
    "HTTPBackend",
    "LocalReadOnlyBackend",
    "ModelReferenceBackend",
    "RedisBackend",
    "ReplicaBackendBase",
]


def __getattr__(name: str) -> type:
    if name == "RedisBackend":
        from .redis_backend import RedisBackend

        return RedisBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
