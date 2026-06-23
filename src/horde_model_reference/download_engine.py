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
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import requests
from loguru import logger

from horde_model_reference.on_disk_layout import file_paths_for

if TYPE_CHECKING:
    from horde_model_reference.model_reference_records import DownloadRecord, GenericModelRecord

__all__ = [
    "R2_BY_HASH_PREFIX",
    "DownloadAborted",
    "DownloadOutcome",
    "ProgressCallback",
    "download_addressed_file",
    "download_file",
    "download_record_files",
    "gateway_accepts_key",
    "gateway_url_for",
    "md5_of",
    "sha256_of",
]

ProgressCallback = Callable[[int, int], None]
"""Called with ``(downloaded_bytes, total_bytes)`` after each written chunk.

May raise :class:`DownloadAborted` to terminate an in-flight download (e.g. on shutdown or config removal);
the engine propagates it out of :func:`download_file` and cleans up the partial file."""

UNKNOWN_SHA256_SENTINEL = "FIXME"
"""The ``DownloadRecord.sha256sum`` value meaning "no known hash"; such files are accepted unverified."""

R2_BY_HASH_PREFIX = "by-hash"
"""URL path prefix under the R2 gateway for content-addressed objects: ``<gateway>/by-hash/<sha256>``."""

_CHUNK_SIZE = 16 * 1024 * 1024
_HASH_CHUNK_SIZE = 2**20
_DEFAULT_MAX_RETRIES = 5
_RETRY_SLEEP_SECONDS = 2.0
_GET_TIMEOUT_SECONDS = 20

_NON_RETRIABLE_STATUSES = frozenset({401, 403, 404, 410})
"""HTTP statuses that are a definitive "no" for a given URL: do not retry, fall back to the next source.

Used so a gated mirror that rejects the key (401/403) or has not yet been populated (404/410) yields
immediately to the origin URL instead of burning the transient-failure retry budget against it."""

_FALLBACK_SOURCE_MAX_RETRIES = 1
"""Retries granted to a non-final candidate (e.g. the gated mirror) before falling through to the next source.

The mirror is strictly an accelerator, so a transient mirror failure (a 5xx or a timeout, neither of which is in
:data:`_NON_RETRIABLE_STATUSES`) must hand off to the origin *promptly* rather than retrying five times with
backoff first: otherwise a degraded mirror would make every download slower than having no mirror at all. The
final candidate (the origin) keeps the full retry budget, since there is nowhere left to fall through to."""

_SEGMENT_THRESHOLD_BYTES = 64 * 1024 * 1024
"""Files smaller than this are never segmented: the per-connection overhead is not worth it, and aux/config
files (which dominate the small end) stay on the simple, resumable single-stream path."""

_MIN_SEGMENT_BYTES = 16 * 1024 * 1024
"""Never cut a file into pieces smaller than this; bounds the segment count for mid-sized files."""

_MAX_CONNECTIONS_PER_FILE = 8
"""Hard ceiling on concurrent connections to one file, regardless of the configured request, so a
misconfiguration cannot open an abusive number of sockets to a single host."""


class DownloadAborted(Exception):
    """Raised out of :func:`download_file` when a progress callback aborts an in-flight download.

    The progress callback signals termination by raising this; the engine catches it, stops every segment,
    removes the partial file, and re-raises so the caller sees a single, uniform abort rather than a torn
    download left half-written on disk.
    """


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


_KEY_SAFE_HTTP_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
"""Hosts for which a plaintext ``http`` gateway is tolerated (local ``wrangler dev`` only)."""


def gateway_accepts_key(gateway_base_url: str) -> bool:
    """Return whether the apikey may be sent to *gateway_base_url* (it must not travel over plaintext).

    The gateway is sent the worker's real AI Horde key (it has to, to validate via ``/v2/find_user``), so the key
    is only ever attached to an ``https`` endpoint. A plaintext ``http`` gateway is refused outright (the mirror
    is then simply skipped and the origin used), except for a localhost dev endpoint where there is no network to
    intercept. This is the universal guard at the point the key header would be attached.
    """
    parsed = urlparse(gateway_base_url)
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and parsed.hostname in _KEY_SAFE_HTTP_HOSTS


def gateway_url_for(gateway_base_url: str, sha256: str) -> str:
    """Return the content-addressed gateway URL for a file with the given *sha256*.

    The object is addressed purely by its hash (``<gateway>/by-hash/<sha256>``), so the same physical file is
    shared across every record that references it regardless of category or per-record file name. The Cloudflare
    Worker gateway and the devops upload tool derive the same key from this one convention.
    """
    return f"{gateway_base_url.rstrip('/')}/{R2_BY_HASH_PREFIX}/{sha256.lower()}"


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


def _partial_is_complete(url: str, partial_size: int, headers: dict[str, str] | None) -> bool:
    """Return whether an existing *partial_size* already equals the server's full Content-Length.

    *headers* carries the same authentication as the real request (e.g. the gateway ``apikey``); without it a
    gated source answers 401 and the probe would wrongly conclude the partial is incomplete and discard it.
    """
    if not partial_size:
        return False
    with requests.get(url, stream=True, headers=headers, allow_redirects=True, timeout=_GET_TIMEOUT_SECONDS) as probe:
        if not probe.ok:
            return False
        remote_size = int(probe.headers.get("Content-Length", 0))
    return remote_size > 0 and partial_size == remote_size


class _SegmentFailed(Exception):
    """A single segment could not be fetched after its retries.

    The whole segmented attempt then falls back to the single-stream path. Distinct from a callback abort,
    which propagates out unchanged instead.
    """


def _total_from_content_range(content_range: str | None) -> int | None:
    """Parse the total size from a ``Content-Range: bytes 0-0/12345`` header, or None when it is unknown."""
    if not content_range:
        return None
    _, _, total_part = content_range.partition("/")
    total_part = total_part.strip()
    if not total_part or total_part == "*":
        return None
    try:
        return int(total_part)
    except ValueError:
        return None


def _probe_range_support(url: str, extra_headers: dict[str, str] | None) -> tuple[bool, int | None]:
    """Return ``(server_honours_range, total_bytes)`` from a one-byte ranged GET.

    A ``206`` with a parseable ``Content-Range`` total means the file can be segmented; the one-byte body is
    discarded. Anything else (notably a ``200``: the server ignored the range and is streaming the whole body)
    means it cannot, and the caller drops to the single-stream path, which keeps the full retry budget and so
    surfaces any genuine error exactly as it does today. *extra_headers* (e.g. a gateway ``apikey``) are sent
    so a gated source is probed under the same authentication it will be fetched with.
    """
    headers = dict(extra_headers) if extra_headers else {}
    headers["Range"] = "bytes=0-0"
    with requests.get(
        url,
        stream=True,
        headers=headers,
        allow_redirects=True,
        timeout=_GET_TIMEOUT_SECONDS,
    ) as probe:
        if probe.status_code == 206:
            return True, _total_from_content_range(probe.headers.get("Content-Range"))
        return False, None


def _plan_segments(total: int, connections: int) -> list[tuple[int, int]]:
    """Split ``0..total-1`` into contiguous ``[start, end]`` byte ranges for up to *connections* threads.

    The count is clamped by the hard ceiling and by :data:`_MIN_SEGMENT_BYTES` so a mid-sized file is not cut
    into slivers. The final segment absorbs the remainder so the ranges exactly tile the file.
    """
    count = max(1, min(connections, _MAX_CONNECTIONS_PER_FILE, total // _MIN_SEGMENT_BYTES))
    if count <= 1:
        return [(0, total - 1)]
    base = total // count
    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        end = total - 1 if index == count - 1 else start + base - 1
        segments.append((start, end))
        start = end + 1
    return segments


def _segmented_download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None,
    progress_callback: ProgressCallback | None,
    extra_headers: dict[str, str] | None,
    connections: int,
    max_retries: int,
) -> DownloadOutcome | None:
    """Fetch *url* into *destination* over several concurrent ranged connections; return the outcome or None.

    Returns a :class:`DownloadOutcome` when the segmented path handled the download (success or a finalized
    result). Returns ``None`` to mean "not applicable, use the single-stream path": the server ignored the
    range probe, the size is unknown or below the segmentation threshold, or a segment failed transiently
    (in which case the sparse partial is discarded first). Any exception raised by *progress_callback*
    (a caller's abort) propagates out unchanged after every segment is stopped and the partial removed.

    Per the Option A resume policy, segmented downloads do not resume: the sparse ``.spart`` is written fresh
    each attempt (and removed on abort/failure), while the resumable contiguous ``.part`` belongs solely to
    the single-stream path so the two never read each other's partial.
    """
    try:
        supports_range, total = _probe_range_support(url, extra_headers)
    except requests.RequestException:
        return None

    if not supports_range or total is None or total < _SEGMENT_THRESHOLD_BYTES:
        return None
    segments = _plan_segments(total, connections)
    if len(segments) <= 1:
        return None

    spart = Path(f"{destination}.spart")
    spart.unlink(missing_ok=True)
    with spart.open("wb") as handle:
        handle.truncate(total)

    abort = threading.Event()
    progress_lock = threading.Lock()
    downloaded_total = 0

    def report(delta: int) -> None:
        nonlocal downloaded_total
        with progress_lock:
            downloaded_total += delta
            snapshot = downloaded_total
        # Called outside the lock; may raise the caller's abort exception, which propagates as this
        # segment's future result and is re-raised unchanged by the aggregation below.
        if progress_callback is not None:
            progress_callback(snapshot, total)

    def fetch_segment(start: int, end: int) -> None:
        attempts_remaining = max_retries
        position = start
        while position <= end:
            if abort.is_set():
                return
            headers = dict(extra_headers) if extra_headers else {}
            headers["Range"] = f"bytes={position}-{end}"
            try:
                with requests.get(
                    url,
                    stream=True,
                    headers=headers,
                    allow_redirects=True,
                    timeout=_GET_TIMEOUT_SECONDS,
                ) as response:
                    if response.status_code != 206:
                        # The probe saw range support but this request was not honoured (or errored): treat
                        # as a segment failure so the whole download retreats to the single-stream path.
                        raise _SegmentFailed
                    with spart.open("r+b") as handle:
                        handle.seek(position)
                        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                            if abort.is_set():
                                return
                            if not chunk:
                                continue
                            handle.write(chunk)
                            position += len(chunk)
                            report(len(chunk))
                    return
            except requests.RequestException:
                attempts_remaining -= 1
                if attempts_remaining <= 0:
                    raise _SegmentFailed from None
                time.sleep(_RETRY_SLEEP_SECONDS)

    with ThreadPoolExecutor(max_workers=len(segments)) as pool:
        futures = [pool.submit(fetch_segment, start, end) for start, end in segments]
        for future in as_completed(futures):
            if future.exception() is not None:
                # One segment failed (or the callback aborted); stop the rest. The pool join on context
                # exit guarantees every thread has finished before we inspect the file below.
                abort.set()

    exceptions = [exc for future in futures if (exc := future.exception()) is not None]
    abort_exc = next((exc for exc in exceptions if not isinstance(exc, _SegmentFailed)), None)
    if abort_exc is not None:
        spart.unlink(missing_ok=True)
        raise abort_exc
    if exceptions:
        spart.unlink(missing_ok=True)
        logger.warning("Segmented download failed; falling back to single-stream: file={}", destination.name)
        return None
    return _finalize(spart, destination, total, expected_sha256)


def download_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    progress_callback: ProgressCallback | None = None,
    auth_query_token: str | None = None,
    extra_headers: dict[str, str] | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    connections: int = 1,
) -> DownloadOutcome:
    """Download *url* to *destination*, resuming a prior ``.part`` when the server supports it.

    The file streams into ``<destination>.part`` (resumed via a ``Range`` request when a partial exists),
    then is atomically renamed onto *destination*. A server that ignores the range request triggers a clean
    restart (the partial is discarded). After a successful write the sha256 is computed (refreshing the
    sidecar cache) and, when *expected_sha256* is a real hash, compared.

    A definitive rejection (HTTP 401/403/404/410) returns a failed outcome immediately without consuming the
    retry budget, so a caller iterating multiple sources (e.g. a gated mirror then its origin) falls through
    promptly rather than hammering a URL that will never serve it.

    When *connections* is greater than 1 and the file is large enough and the server honours range requests,
    the bytes are fetched over that many concurrent ranged connections to raise throughput on a single file
    (a single TCP stream is often window/RTT-limited well below the link). This fast path does not resume an
    interrupted download; on any unsuitability (small file, no range support, a transient segment failure) it
    transparently drops to the single-stream behaviour below. ``connections == 1`` is exactly the original
    single-stream path.

    Args:
        url: The file URL.
        destination: The final on-disk path. Parent directories are created as needed.
        expected_sha256: The record's declared hash, or None / ``"FIXME"`` to skip verification.
        progress_callback: Optional ``(downloaded_bytes, total_bytes)`` callback, invoked per chunk. May raise
            to abort the download (see :class:`DownloadAborted`); the abort propagates out unchanged.
        auth_query_token: Optional token appended to the URL query (e.g. a CivitAI API token).
        extra_headers: Optional request headers (e.g. an ``apikey`` header for the gated R2 gateway). Merged
            with the resume ``Range`` header; a ``Range`` key here is overridden when resuming.
        max_retries: Maximum transient-failure retries before giving up.
        connections: Maximum concurrent connections for a single large file (clamped to
            :data:`_MAX_CONNECTIONS_PER_FILE`); 1 keeps the simple resumable single-stream path.

    Returns:
        A :class:`DownloadOutcome` describing the result.
    """
    destination = Path(destination)
    partial = Path(f"{destination}.part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    download_url = _with_auth_token(url, auth_query_token)

    if connections > 1:
        segmented = _segmented_download(
            download_url,
            destination,
            expected_sha256=expected_sha256,
            progress_callback=progress_callback,
            extra_headers=extra_headers,
            connections=connections,
            max_retries=max_retries,
        )
        if segmented is not None:
            return segmented

    attempts_remaining = max_retries
    while attempts_remaining > 0:
        partial_size = partial.stat().st_size if partial.exists() else 0
        headers = dict(extra_headers) if extra_headers else {}
        if partial_size:
            headers["Range"] = f"bytes={partial_size}-"
        try:
            with requests.get(
                download_url,
                stream=True,
                headers=headers,
                allow_redirects=True,
                timeout=_GET_TIMEOUT_SECONDS,
            ) as response:
                if response.status_code in _NON_RETRIABLE_STATUSES:
                    logger.debug(
                        "Source declined {} ({}); not retrying this source",
                        destination.name,
                        response.status_code,
                    )
                    return DownloadOutcome(success=False, final_path=destination, bytes_written=0, sha256=None)

                if response.status_code == 416:
                    if _partial_is_complete(download_url, partial_size, extra_headers):
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


def download_addressed_file(
    origin_url: str,
    destination: Path,
    *,
    sha256: str | None,
    gateway_base_url: str | None = None,
    apikey: str | None = None,
    auth_query_token: str | None = None,
    progress_callback: ProgressCallback | None = None,
    use_configured_gateway: bool = True,
    connections: int = 1,
) -> DownloadOutcome:
    """Fetch one file, preferring the gated content-addressed R2 gateway and falling back to *origin_url*.

    This is the record-free entry point: callers that have a plain ``(origin_url, sha256)`` rather than a
    :class:`DownloadRecord` (e.g. the controlnet-annotator prefetch) get the identical mirror-first behaviour.

    The gateway is attempted only when it can actually serve the file: a gateway base URL and an apikey are both
    configured and a real *sha256* (the content address) is known. Any gateway failure (an ineligible/invalid
    key, an object not yet mirrored, a 5xx/timeout, or a network error) transparently falls through to
    *origin_url*, so an absent or unreachable mirror never blocks a download. The sha256 is verified after every
    attempt regardless of source, so a mismatched mirror object is rejected; a completed-but-corrupt attempt is
    removed before the next source is tried so a later failure cannot leave a bad file behind.

    *connections* applies only to the *origin* fetch: the gateway is the already-fast accelerator and is left
    single-stream until its Cloudflare Worker is confirmed to pass ``Range`` through to R2 (segmenting a server
    that buffers the range would help nothing and waste connections).
    """
    if gateway_base_url is None and use_configured_gateway:
        from horde_model_reference import HordeModelReferenceSettings

        gateway_base_url = HordeModelReferenceSettings().r2.gateway_url

    candidates: list[tuple[str, dict[str, str] | None, str | None, int]] = []
    have_address = bool(sha256) and sha256 != UNKNOWN_SHA256_SENTINEL
    if gateway_base_url and apikey and have_address and gateway_accepts_key(gateway_base_url):
        candidates.append((gateway_url_for(gateway_base_url, sha256), {"apikey": apikey}, None, 1))
    candidates.append((origin_url, None, auth_query_token, connections))

    outcome = DownloadOutcome(success=False, final_path=destination, bytes_written=0, sha256=None)
    for index, (candidate_url, candidate_headers, candidate_token, candidate_connections) in enumerate(candidates):
        is_last = index == len(candidates) - 1
        outcome = download_file(
            candidate_url,
            destination,
            expected_sha256=sha256,
            progress_callback=progress_callback,
            auth_query_token=candidate_token,
            extra_headers=candidate_headers,
            max_retries=_DEFAULT_MAX_RETRIES if is_last else _FALLBACK_SOURCE_MAX_RETRIES,
            connections=candidate_connections,
        )
        if outcome.success:
            return outcome
        # A completed-but-corrupt attempt (sha256 computed but mismatched) has already replaced the
        # destination; remove the bad file so it is never trusted later (a present file is otherwise skipped
        # on the next run). This applies even to the final candidate. A transient failure (sha256 is None)
        # instead keeps its ``.part`` so the next source, or the next run, can resume it.
        if outcome.sha256 is not None:
            destination.unlink(missing_ok=True)
            Path(f"{destination}.part").unlink(missing_ok=True)
    return outcome


def _fetch_file_with_fallback(
    download: DownloadRecord,
    destination: Path,
    *,
    gateway_base_url: str | None,
    apikey: str | None,
    auth_query_token: str | None,
    progress_callback: ProgressCallback | None,
    use_configured_gateway: bool,
    connections: int,
) -> DownloadOutcome:
    """Fetch one declared file, preferring the gated R2 gateway and falling back to its origin ``file_url``.

    A thin :class:`DownloadRecord` adapter over :func:`download_addressed_file`.
    """
    return download_addressed_file(
        download.file_url,
        destination,
        sha256=download.sha256sum,
        gateway_base_url=gateway_base_url,
        apikey=apikey,
        auth_query_token=auth_query_token,
        progress_callback=progress_callback,
        use_configured_gateway=use_configured_gateway,
        connections=connections,
    )


def download_record_files(
    record: GenericModelRecord,
    root: Path,
    *,
    progress_callback: ProgressCallback | None = None,
    auth_query_token: str | None = None,
    gateway_base_url: str | None = None,
    apikey: str | None = None,
    use_configured_gateway: bool = True,
    connections: int = 1,
) -> bool:
    """Download every file *record* declares into its category folder under *root*.

    Files already on disk are kept when they still satisfy their declared sha256 (their sidecar is ensured for
    discovery); one that fails the hash is removed and re-fetched. Missing files are fetched via
    :func:`download_file`. Always targets the primary *root*: discovery may search extra directories, but new
    downloads land deterministically under *root*.

    When *gateway_base_url* (or the configured ``r2.gateway_url``) and *apikey* are both supplied, each file is
    first attempted from the gated, content-addressed R2 gateway and falls back to its origin URL on any failure
    (see :func:`_fetch_file_with_fallback`). Omitting an apikey (or a record with no real sha256) downloads
    straight from the origin URL, preserving the prior behaviour for anonymous and standalone callers.

    Args:
        record: The model record whose declared files should be present.
        root: The model-weights root under which the category folder lives.
        progress_callback: Optional ``(downloaded_bytes, total_bytes)`` callback, forwarded per file.
        auth_query_token: Optional token appended to each origin file URL's query (e.g. a CivitAI token).
        gateway_base_url: Optional base URL of the gated R2 gateway; enables the mirror-first path.
        apikey: Optional horde API key sent to the gateway; required for the mirror-first path.
        use_configured_gateway: When True, use ``HordeModelReferenceSettings().r2.gateway_url`` if
            *gateway_base_url* is not supplied.
        connections: Maximum concurrent connections per large file, forwarded to the origin fetch in
            :func:`download_addressed_file` (the gateway stays single-stream).

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
            # Ensure the sidecar (for discovery) and re-validate against the declared hash: a file left behind
            # by a past corrupt download would otherwise be trusted forever simply because it exists. An
            # unknown/``FIXME`` hash matches by definition, preserving the unverified-by-design behaviour.
            if _checksum_matches(sha256_of(destination), download.sha256sum):
                continue
            logger.warning("On-disk file fails its declared sha256; re-downloading: file={}", destination.name)
            destination.unlink(missing_ok=True)
        outcome = _fetch_file_with_fallback(
            download,
            destination,
            gateway_base_url=gateway_base_url,
            apikey=apikey,
            auth_query_token=auth_query_token,
            progress_callback=progress_callback,
            use_configured_gateway=use_configured_gateway,
            connections=connections,
        )
        if not outcome.success:
            all_succeeded = False
    return all_succeeded
