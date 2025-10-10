"""Backend providers for model reference data sources.

This module defines the abstract backend interface and concrete implementations
for fetching model reference data from different sources:

- FileSystemBackend: PRIMARY mode - reads/writes local JSON files
- GitHubBackend: REPLICA mode - downloads from GitHub with legacy conversion
- HTTPBackend: REPLICA mode - fetches from PRIMARY API with GitHub fallback
- RedisBackend: PRIMARY mode wrapper - adds distributed caching

Each backend is designed for specific replication modes and use cases.
"""

from .base import ModelReferenceBackend
from .filesystem_backend import FileSystemBackend
from .github_backend import GitHubBackend
from .http_backend import HTTPBackend
from .redis_backend import RedisBackend
from .replica_backend_base import ReplicaBackendBase

__all__ = [
    "FileSystemBackend",
    "GitHubBackend",
    "HTTPBackend",
    "ModelReferenceBackend",
    "RedisBackend",
    "ReplicaBackendBase",
]
