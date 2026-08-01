"""The object-store seam the R2 sync tool writes through: a tiny put/head surface over a bucket.

The sync tool only ever needs to ask "is this content-addressed object already present?" and "upload these bytes
under this key". Narrowing the dependency to those two operations keeps the tool testable without a live bucket
(:class:`InMemoryObjectStore`) and keeps the boto3 import lazy so the rest of the package, and the test suite,
do not require boto3 to be installed.

Keys follow the engine's single content-addressing convention via :func:`object_key_for`, so the uploader, the
Cloudflare Worker gateway, and the download engine all derive the same key from a file's sha256.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from horde_model_reference.download_engine import R2_BY_HASH_PREFIX

if TYPE_CHECKING:
    from horde_model_reference import R2Settings

__all__ = [
    "InMemoryObjectStore",
    "ObjectInfo",
    "ObjectStore",
    "R2ObjectStore",
    "object_key_for",
]


@dataclass(frozen=True)
class ObjectInfo:
    """Represent the bucket metadata needed to reconcile a content-addressed object."""

    size_bytes: int
    """Stored object size in bytes."""
    metadata: dict[str, str]
    """Custom metadata stored with the object."""


def object_key_for(sha256: str) -> str:
    """Return the content-addressed object key for a file with the given *sha256* (``by-hash/<sha256>``).

    The single source of the bucket layout: the same key the gateway resolves and the download engine derives
    from :func:`horde_model_reference.download_engine.gateway_url_for`.
    """
    return f"{R2_BY_HASH_PREFIX}/{sha256.lower()}"


class ObjectStore(Protocol):
    """The minimal bucket surface the sync tool needs: existence checks and content-addressed uploads."""

    def head(self, key: str) -> ObjectInfo | None:
        """Return metadata for *key*, or ``None`` when it does not exist."""
        ...

    def put(
        self,
        key: str,
        source_path: Path,
        *,
        metadata: dict[str, str],
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload the bytes at *source_path* to *key*, attaching *metadata* (provenance/license) to the object."""
        ...


class InMemoryObjectStore:
    """An in-memory :class:`ObjectStore` for tests and dry-run planning, recording keys, sizes and metadata."""

    def __init__(self, *, present: set[str] | None = None) -> None:
        """Start with an optional set of keys treated as already *present* in the bucket."""
        self.objects: dict[str, dict[str, str]] = {
            key: {"sha256": key.rsplit("/", maxsplit=1)[-1]} for key in (present or set())
        }
        self.sizes: dict[str, int] = dict.fromkeys(present or set(), 0)
        self.content_types: dict[str, str] = {}

    def head(self, key: str) -> ObjectInfo | None:
        """Return metadata for a key that has been put or pre-seeded."""
        if key not in self.objects:
            return None
        return ObjectInfo(size_bytes=self.sizes[key], metadata=dict(self.objects[key]))

    def put(
        self,
        key: str,
        source_path: Path,
        *,
        metadata: dict[str, str],
        content_type: str = "application/octet-stream",
    ) -> None:
        """Record *key* with its byte size and *metadata*, as a real upload would."""
        self.objects[key] = dict(metadata)
        self.sizes[key] = source_path.stat().st_size
        self.content_types[key] = content_type


class R2ObjectStore:
    """A boto3-backed :class:`ObjectStore` for a Cloudflare R2 bucket (S3-compatible API).

    boto3 is imported lazily so the package and its tests do not depend on it; only the maintainer running an
    actual upload needs it installed (the ``r2`` dependency group).
    """

    def __init__(self, settings: R2Settings) -> None:
        """Build an S3 client for the R2 bucket described by *settings*.

        Raises:
            RuntimeError: If the bucket or endpoint is not configured, or boto3 is not installed.
        """
        if not settings.upload_bucket or not settings.upload_endpoint_url:
            raise RuntimeError(
                "R2 upload requires HORDE_MODEL_REFERENCE_R2__UPLOAD_BUCKET and "
                "HORDE_MODEL_REFERENCE_R2__UPLOAD_ENDPOINT_URL to be set.",
            )
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise RuntimeError("R2 upload needs boto3; install the 'r2' dependency group.") from exc

        self._bucket = settings.upload_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.upload_endpoint_url,
            aws_access_key_id=settings.upload_access_key_id,
            aws_secret_access_key=settings.upload_secret_access_key,
            region_name=settings.upload_region,
        )

    def head(self, key: str) -> ObjectInfo | None:
        """Return metadata for *key* (a 404 from ``head_object`` means absent)."""
        from botocore.exceptions import ClientError

        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        return ObjectInfo(
            size_bytes=int(response["ContentLength"]),
            metadata={
                str(metadata_key): str(metadata_value)
                for metadata_key, metadata_value in response.get("Metadata", {}).items()
            },
        )

    def put(
        self,
        key: str,
        source_path: Path,
        *,
        metadata: dict[str, str],
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload *source_path* to *key* with object *metadata* (object metadata values must be strings)."""
        self._client.upload_file(
            str(source_path),
            self._bucket,
            key,
            ExtraArgs={
                "ContentType": content_type,
                "Metadata": {metadata_key: str(metadata_value) for metadata_key, metadata_value in metadata.items()},
            },
        )
