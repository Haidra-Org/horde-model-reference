"""Torch-free model-file download engine: resumable HTTP fetch and checksum sidecars.

The single place that knows how to fetch the files a model record declares: a resumable streaming download
with retry and atomic ``.part`` rename, plus mtime-keyed sha256/md5 sidecar caching. Owned by
:mod:`horde_model_reference` so every consumer (the worker download process, hordelib discovery managers,
third-party tools) shares one implementation instead of re-deriving it.

Progress is reported purely through an optional ``(downloaded_bytes, total_bytes)`` callback; no progress-bar
or UI dependency is pulled in.

Security note: a record whose ``sha256sum`` is the unknown-checksum sentinel (``"FIXME"``) is accepted
without verification, matching long-standing behaviour. This is a deliberate gap, not an oversight: many
reference entries still lack a real hash. It is preserved here rather than silently changed.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from loguru import logger

from horde_model_reference.on_disk_layout import file_paths_for

if TYPE_CHECKING:
    from horde_model_reference.model_reference_records import GenericModelRecord

__all__ = [
    "DownloadOutcome",
    "ProgressCallback",
    "download_file",
    "download_record_files",
    "md5_of",
    "sha256_of",
]

ProgressCallback = Callable[[int, int], None]
"""Called with ``(downloaded_bytes, total_bytes)`` after each written chunk."""

UNKNOWN_SHA256_SENTINEL = "FIXME"
"""The ``DownloadRecord.sha256sum`` value meaning "no known hash"; such files are accepted unverified."""

_CHUNK_SIZE = 16 * 1024 * 1024
_HASH_CHUNK_SIZE = 2**20
_DEFAULT_MAX_RETRIES = 5
_RETRY_SLEEP_SECONDS = 2.0
_GET_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class DownloadOutcome:
    """The result of fetching a single file."""

    success: bool
    """Whether the file is now on disk and (when a hash was known) matched it."""
    final_path: Path
    """The resolved on-disk path the file was written to."""
    bytes_written: int
    """Total bytes present in the finished file (including any resumed prefix)."""
    sha256: str | None
    """The computed sha256 of the finished file, or None when the download did not complete."""


def _hash_file(path: Path, algorithm: str) -> str:
    """Return the hex digest of *path*, read in chunks, using the named hashlib *algorithm*."""
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_sidecar(sidecar: Path, digest: str, file_name: str) -> None:
    """Write a ``<hash> *<file_name>`` sidecar, ignoring write failures (e.g. read-only volume)."""
    try:
        sidecar.write_text(f"{digest} *{file_name}")
    except OSError:
        logger.debug("Could not write checksum sidecar: sidecar={}", sidecar)


def _cached_digest(path: Path, sidecar: Path) -> str | None:
    """Return the sidecar's cached digest when it is newer than *path*, else None."""
    if sidecar.is_file() and sidecar.stat().st_mtime > path.stat().st_mtime:
        return sidecar.read_text().split()[0]
    return None


def sha256_of(path: Path) -> str:
    """Return the sha256 of *path*, using (and refreshing) a ``.sha256`` sidecar cache.

    The cache is keyed on modification time: a sidecar newer than the file is trusted; otherwise the hash is
    recomputed and the sidecar rewritten. Writing the sidecar is best-effort.

    Raises:
        FileNotFoundError: If *path* is not an existing file.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No file {path}")
    sidecar = path.with_suffix(".sha256")
    cached = _cached_digest(path, sidecar)
    if cached is not None:
        return cached
    logger.info("Calculating sha256sum (this may take a while): file={}", path.name)
    digest = _hash_file(path, "sha256")
    _write_sidecar(sidecar, digest, path.name)
    return digest


def md5_of(path: Path) -> str | None:
    """Return the md5 of *path* (using a ``.md5`` sidecar cache), or None when *path* is not a file."""
    if not path.is_file():
        return None
    sidecar = path.with_suffix(".md5")
    cached = _cached_digest(path, sidecar)
    if cached is not None:
        return cached
    digest = _hash_file(path, "md5")
    _write_sidecar(sidecar, digest, path.name)
    return digest


def _with_auth_token(url: str, token: str | None) -> str:
    """Return *url* with ``token=<token>`` appended to the query string, when a token is given."""
    if not token:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}token={token}"


def _checksum_matches(actual_sha256: str, expected_sha256: str | None) -> bool:
    """Return whether *actual_sha256* satisfies *expected_sha256* (unknown/absent hashes always pass)."""
    if expected_sha256 is None or expected_sha256 == UNKNOWN_SHA256_SENTINEL:
        return True
    return actual_sha256.lower() == expected_sha256.lower()


def _finalize(partial: Path, destination: Path, bytes_written: int, expected_sha256: str | None) -> DownloadOutcome:
    """Atomically move *partial* onto *destination*, then hash and (when known) validate it."""
    os.replace(partial, destination)
    digest = sha256_of(destination)
    matched = _checksum_matches(digest, expected_sha256)
    if not matched:
        logger.error(
            "Checksum mismatch for {}: expected={}, actual={}",
            destination.name,
            expected_sha256,
            digest,
        )
    return DownloadOutcome(success=matched, final_path=destination, bytes_written=bytes_written, sha256=digest)


def _partial_is_complete(url: str, partial_size: int) -> bool:
    """Return whether an existing *partial_size* already equals the server's full Content-Length."""
    if not partial_size:
        return False
    with requests.get(url, stream=True, allow_redirects=True, timeout=_GET_TIMEOUT_SECONDS) as probe:
        remote_size = int(probe.headers.get("Content-Length", 0))
    return remote_size > 0 and partial_size == remote_size


def download_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    progress_callback: ProgressCallback | None = None,
    auth_query_token: str | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> DownloadOutcome:
    """Download *url* to *destination*, resuming a prior ``.part`` when the server supports it.

    The file streams into ``<destination>.part`` (resumed via a ``Range`` request when a partial exists),
    then is atomically renamed onto *destination*. A server that ignores the range request triggers a clean
    restart (the partial is discarded). After a successful write the sha256 is computed (refreshing the
    sidecar cache) and, when *expected_sha256* is a real hash, compared.

    Args:
        url: The file URL.
        destination: The final on-disk path. Parent directories are created as needed.
        expected_sha256: The record's declared hash, or None / ``"FIXME"`` to skip verification.
        progress_callback: Optional ``(downloaded_bytes, total_bytes)`` callback, invoked per chunk.
        auth_query_token: Optional token appended to the URL query (e.g. a CivitAI API token).
        max_retries: Maximum transient-failure retries before giving up.

    Returns:
        A :class:`DownloadOutcome` describing the result.
    """
    destination = Path(destination)
    partial = Path(f"{destination}.part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    download_url = _with_auth_token(url, auth_query_token)

    attempts_remaining = max_retries
    while attempts_remaining > 0:
        partial_size = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={partial_size}-"} if partial_size else {}
        try:
            with requests.get(
                download_url,
                stream=True,
                headers=headers,
                allow_redirects=True,
                timeout=_GET_TIMEOUT_SECONDS,
            ) as response:
                if response.status_code == 416:
                    if _partial_is_complete(download_url, partial_size):
                        return _finalize(partial, destination, partial_size, expected_sha256)
                    partial.unlink(missing_ok=True)
                    continue

                if partial_size and response.status_code != 206:
                    content_length = int(response.headers.get("Content-Length", 0)) if response.ok else 0
                    if content_length and partial_size == content_length:
                        return _finalize(partial, destination, partial_size, expected_sha256)
                    logger.warning(
                        "Server ignored resume request ({}); restarting download of {}",
                        response.status_code,
                        destination.name,
                    )
                    partial.unlink(missing_ok=True)
                    continue

                if not response.ok:
                    response.raise_for_status()

                content_length = int(response.headers.get("Content-Length", 0))
                total = content_length + partial_size
                downloaded = partial_size
                with partial.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback is not None:
                            progress_callback(downloaded, total)
                return _finalize(partial, destination, downloaded, expected_sha256)

        except requests.RequestException:
            attempts_remaining -= 1
            if attempts_remaining <= 0:
                logger.info("Download failed after retries: file={}", destination.name)
                return DownloadOutcome(success=False, final_path=destination, bytes_written=0, sha256=None)
            logger.info("Retrying download: file={}", destination.name)
            time.sleep(_RETRY_SLEEP_SECONDS)

    return DownloadOutcome(success=False, final_path=destination, bytes_written=0, sha256=None)


def download_record_files(
    record: GenericModelRecord,
    root: Path,
    *,
    progress_callback: ProgressCallback | None = None,
    auth_query_token: str | None = None,
) -> bool:
    """Download every file *record* declares into its category folder under *root*.

    Files already on disk are skipped (their sha256 sidecar is ensured for discovery); missing files are
    fetched via :func:`download_file`. Always targets the primary *root*: discovery may search extra
    directories, but new downloads land deterministically under *root*.

    Args:
        record: The model record whose declared files should be present.
        root: The model-weights root under which the category folder lives.
        progress_callback: Optional ``(downloaded_bytes, total_bytes)`` callback, forwarded per file.
        auth_query_token: Optional token appended to each file URL's query.

    Returns:
        True when every declared file is present (and any known hash matched), False otherwise.
    """
    destinations = file_paths_for(record, Path(root))
    if not destinations:
        logger.warning("No resolvable download targets for record: name={}", record.name)
        return False

    all_succeeded = True
    for download, destination in zip(record.config.download, destinations, strict=True):
        if destination.exists():
            sha256_of(destination)
            continue
        outcome = download_file(
            download.file_url,
            destination,
            expected_sha256=download.sha256sum,
            progress_callback=progress_callback,
            auth_query_token=auth_query_token,
        )
        if not outcome.success:
            all_succeeded = False
    return all_succeeded
