"""Supply the bytes of a declared file, preferring a local mirror and falling back to the origin host.

The upload tool needs the actual bytes to hash and to upload, but most files it processes will already sit in
the maintainer's local model cache. So this looks there first (resolving the canonical on-disk path the same
way the worker does) and only downloads from the record's origin URL when no local copy exists, caching the
fetch so a re-run does not refetch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from horde_model_reference.download_engine import UNKNOWN_SHA256_SENTINEL, download_file, sha256_of
from horde_model_reference.on_disk_layout import file_paths_for

if TYPE_CHECKING:
    from collections.abc import Sequence

    from horde_model_reference.model_reference_records import DownloadRecord, GenericModelRecord

__all__ = ["LocalThenOriginByteSource"]

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _known_sha(value: str | None) -> str | None:
    """Return a real expected sha256, or None when the reference does not know it."""
    return value if value and value != UNKNOWN_SHA256_SENTINEL else None


def _safe_cache_name(*, namespace: str, source_id: str, display_name: str) -> Path:
    """Return a cache-relative path that cannot be influenced by path separators in reference data."""
    safe_namespace = _UNSAFE_NAME.sub("_", namespace).strip("._-") or "model"
    safe_display = _UNSAFE_NAME.sub("_", Path(display_name).name).strip("._-") or "file"
    digest = sha256(source_id.encode("utf-8")).hexdigest()[:16]
    return Path(safe_namespace) / f"{digest}-{safe_display}"


@dataclass
class LocalThenOriginByteSource:
    """A :class:`scripts.r2_sync.planner.ByteSource` that reads a local mirror, else fetches from origin."""

    weights_root: Path
    """The model-weights root under which local category folders live."""
    extra_roots: Sequence[Path] = ()
    """Additional weights roots to search for an existing local copy (multi-disk deployments)."""
    cache_dir: Path | None = None
    """Where origin downloads are cached. When None, files not present locally are reported missing."""
    auth_query_token: str | None = None
    """Optional query token for origin hosts that need one (e.g. a CivitAI token)."""

    _origin_fetches: int = field(default=0, init=False)
    """How many origin downloads were performed (for run reporting/tests)."""

    def acquire(self, record: GenericModelRecord, download: DownloadRecord) -> Path | None:
        """Return a local path holding *download*'s bytes, fetching from origin into the cache if needed."""
        local = self._local_copy(record, download)
        if local is not None:
            return local
        if self.cache_dir is None:
            return None

        target = self.cache_dir / _safe_cache_name(
            namespace=record.name,
            source_id=download.file_url,
            display_name=download.file_name,
        )
        expected_sha = _known_sha(download.sha256sum)
        if target.is_file():
            if expected_sha is not None and sha256_of(target).lower() != expected_sha.lower():
                target.unlink()
            else:
                return target

        self._origin_fetches += 1
        outcome = download_file(
            download.file_url,
            target,
            expected_sha256=download.sha256sum,
            auth_query_token=self.auth_query_token,
        )
        if expected_sha is not None and target.is_file() and not outcome.success:
            target.unlink(missing_ok=True)
            Path(f"{target}.part").unlink(missing_ok=True)
        # Unknown-hash downloads are returned for backfill. Known-hash corrupt downloads are removed above so a
        # future run cannot trust poisoned cache bytes.
        return target if target.is_file() else None

    def _local_copy(self, record: GenericModelRecord, download: DownloadRecord) -> Path | None:
        """Return the existing on-disk path for *download* within the local weights roots, if present."""
        index = next((i for i, candidate in enumerate(record.config.download) if candidate is download), None)
        if index is None:
            return None
        paths = file_paths_for(record, self.weights_root, extra_roots=tuple(self.extra_roots))
        if index >= len(paths):
            return None
        local = paths[index]
        return local if local.is_file() else None
