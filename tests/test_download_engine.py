"""Tests for the torch-free download engine, exercised against a real local HTTP server.

The engine uses ``requests``, so the tests drive a small threaded HTTP server that gives full control over
Range/206/416/200 behaviour, exercising the genuine resume and restart state machine rather than a mock.
"""

from __future__ import annotations

import hashlib
import http.server
import os
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import horde_model_reference.download_engine as download_engine
from horde_model_reference.download_engine import (
    download_file,
    download_record_files,
    gateway_accepts_key,
    sha256_of,
)
from horde_model_reference.meta_consts import (
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    ModelClassification,
)
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
)


@dataclass
class _ServerState:
    """Mutable control surface for the test HTTP server."""

    payload: bytes = b""
    mode: str = "normal"  # "normal" | "ignore_range" | "fail_then_ok"
    fail_times: int = 0
    get_count: int = 0
    range_get_count: int = 0
    force_status: int | None = None
    """When set, every GET responds with this status and no body (simulates a gateway 401/403/404)."""
    seen_apikeys: list[str] = field(default_factory=list)
    """The ``apikey`` request header seen on each GET (empty string when absent), for asserting forwarding."""


class _StatefulServer(http.server.ThreadingHTTPServer):
    """An HTTP server carrying a :class:`_ServerState` the handler reads per request."""

    state: _ServerState


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves a fixed payload with configurable Range/failure behaviour."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - matches base signature
        """Silence the test server's per-request logging."""
        return

    def _state(self) -> _ServerState:
        assert isinstance(self.server, _StatefulServer)
        return self.server.state

    def do_HEAD(self) -> None:
        """Respond to HEAD with the payload length."""
        state = self._state()
        self.send_response(200)
        self.send_header("Content-Length", str(len(state.payload)))
        self.end_headers()

    def do_GET(self) -> None:
        """Respond to GET, honouring Range unless configured to ignore or fail."""
        state = self._state()
        state.get_count += 1
        state.seen_apikeys.append(self.headers.get("apikey", ""))

        if state.force_status is not None:
            self.send_response(state.force_status)
            self.end_headers()
            return

        if state.mode == "fail_then_ok" and state.get_count <= state.fail_times:
            self.send_response(500)
            self.end_headers()
            return

        payload = state.payload
        range_header = self.headers.get("Range")
        if range_header and state.mode != "ignore_range":
            state.range_get_count += 1
            start = int(range_header.split("=")[1].split("-")[0])
            if start >= len(payload):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(payload)}")
                self.end_headers()
                return
            remaining = payload[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
            self.send_header("Content-Length", str(len(remaining)))
            self.end_headers()
            self.wfile.write(remaining)
            return

        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def _running_server() -> Iterator[tuple[str, _ServerState]]:
    """Start a local HTTP server and yield ``(base_url, state)``, shutting it down on exit."""
    server = _StatefulServer(("127.0.0.1", 0), _Handler)
    server.state = _ServerState()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}", server.state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def http_server() -> Iterator[tuple[str, _ServerState]]:
    """Yield ``(base_url, state)`` for a running local HTTP server, shut down on teardown."""
    with _running_server() as running:
        yield running


@pytest.fixture
def gateway_and_origin() -> Iterator[tuple[tuple[str, _ServerState], tuple[str, _ServerState]]]:
    """Yield two independent servers ``((gateway_url, gateway_state), (origin_url, origin_state))``.

    Models the gated R2 mirror (first) and the record's origin host (second) so the mirror-first/fallback
    behaviour of :func:`download_record_files` can be exercised end to end.
    """
    with _running_server() as gateway, _running_server() as origin:
        yield gateway, origin


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fresh_download_writes_file_sidecar_and_reports_progress(
    http_server: tuple[str, _ServerState], tmp_path: Path
) -> None:
    """Verify a clean download writes the file, computes the sidecar, leaves no .part, and reports progress."""
    base_url, state = http_server
    state.payload = b"x" * (40 * 1024 * 1024)
    destination = tmp_path / "model.bin"
    seen: list[tuple[int, int]] = []

    outcome = download_file(
        f"{base_url}/model.bin",
        destination,
        expected_sha256=_sha256(state.payload),
        progress_callback=lambda downloaded, total: seen.append((downloaded, total)),
    )

    assert outcome.success is True
    assert destination.read_bytes() == state.payload
    assert destination.with_suffix(".sha256").is_file()
    assert not destination.with_suffix(".bin.part").exists()
    assert seen[-1][0] == len(state.payload)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_checksum_mismatch_reports_failure(http_server: tuple[str, _ServerState], tmp_path: Path) -> None:
    """Verify a download whose bytes do not match the declared hash reports failure."""
    base_url, state = http_server
    state.payload = b"hello world"
    destination = tmp_path / "model.bin"

    outcome = download_file(f"{base_url}/model.bin", destination, expected_sha256="deadbeef")

    assert outcome.success is False
    assert outcome.sha256 == _sha256(state.payload)


def test_unknown_checksum_sentinel_is_accepted(http_server: tuple[str, _ServerState], tmp_path: Path) -> None:
    """Verify the 'FIXME' sentinel accepts the file without verification (preserved security gap)."""
    base_url, state = http_server
    state.payload = b"unverified"
    destination = tmp_path / "model.bin"

    outcome = download_file(f"{base_url}/model.bin", destination, expected_sha256="FIXME")

    assert outcome.success is True


def test_resume_completes_from_partial(http_server: tuple[str, _ServerState], tmp_path: Path) -> None:
    """Verify a download resumes from an existing .part when the server honours Range (206)."""
    base_url, state = http_server
    state.payload = b"abcdefghijklmnopqrstuvwxyz" * 1000
    destination = tmp_path / "model.bin"
    partial = tmp_path / "model.bin.part"
    partial.write_bytes(state.payload[:5000])

    outcome = download_file(f"{base_url}/model.bin", destination, expected_sha256=_sha256(state.payload))

    assert outcome.success is True
    assert destination.read_bytes() == state.payload
    assert state.range_get_count >= 1


def test_already_complete_partial_is_finalized_on_416(http_server: tuple[str, _ServerState], tmp_path: Path) -> None:
    """Verify a fully-downloaded .part is finalized when the server answers the range with 416."""
    base_url, state = http_server
    state.payload = b"complete payload contents"
    destination = tmp_path / "model.bin"
    partial = tmp_path / "model.bin.part"
    partial.write_bytes(state.payload)

    outcome = download_file(f"{base_url}/model.bin", destination, expected_sha256=_sha256(state.payload))

    assert outcome.success is True
    assert destination.read_bytes() == state.payload
    assert not partial.exists()


def test_restart_when_server_ignores_range(http_server: tuple[str, _ServerState], tmp_path: Path) -> None:
    """Verify a stale .part is discarded and the file re-downloaded cleanly when the server ignores Range."""
    base_url, state = http_server
    state.payload = b"0123456789" * 500
    state.mode = "ignore_range"
    destination = tmp_path / "model.bin"
    partial = tmp_path / "model.bin.part"
    partial.write_bytes(b"STALE" * 100)

    outcome = download_file(f"{base_url}/model.bin", destination, expected_sha256=_sha256(state.payload))

    assert outcome.success is True
    # The restart must not append onto the stale partial.
    assert destination.read_bytes() == state.payload


def test_retries_transient_failures_then_succeeds(http_server: tuple[str, _ServerState], tmp_path: Path) -> None:
    """Verify transient 5xx responses are retried until the download succeeds."""
    base_url, state = http_server
    state.payload = b"eventually ok"
    state.mode = "fail_then_ok"
    state.fail_times = 2
    destination = tmp_path / "model.bin"

    outcome = download_file(
        f"{base_url}/model.bin",
        destination,
        expected_sha256=_sha256(state.payload),
        max_retries=5,
    )

    assert outcome.success is True
    assert destination.read_bytes() == state.payload
    assert state.get_count >= 3


def test_sha256_sidecar_cache_is_used_when_newer(tmp_path: Path) -> None:
    """Verify sha256_of trusts a sidecar that is newer than the file (mtime-keyed cache)."""
    target = tmp_path / "weights.bin"
    target.write_bytes(b"real content")
    sidecar = target.with_suffix(".sha256")
    sidecar.write_text("cafef00d *weights.bin")
    # Make the sidecar newer than the file so the cache is trusted.
    file_mtime = target.stat().st_mtime
    os.utime(sidecar, (file_mtime + 10, file_mtime + 10))

    assert sha256_of(target) == "cafef00d"


def _record(downloads: list[DownloadRecord]) -> GenericModelRecord:
    return GenericModelRecord(
        name="m",
        record_type=MODEL_REFERENCE_CATEGORY.miscellaneous,
        model_classification=ModelClassification(domain=MODEL_DOMAIN.image, purpose=MODEL_PURPOSE.miscellaneous),
        config=GenericModelRecordConfig(download=downloads),
    )


def test_download_record_files_places_files_under_category_folder(
    http_server: tuple[str, _ServerState], tmp_path: Path
) -> None:
    """Verify download_record_files writes each declared file into the category folder under the root."""
    base_url, state = http_server
    state.payload = b"weights bytes"
    record = _record(
        [
            DownloadRecord(
                file_name="a.bin",
                file_url=f"{base_url}/a.bin",
                sha256sum=_sha256(state.payload),
            ),
        ],
    )

    assert download_record_files(record, tmp_path) is True
    landed = tmp_path / "miscellaneous" / "a.bin"
    assert landed.read_bytes() == state.payload
    assert landed.with_suffix(".sha256").is_file()


def test_download_record_files_skips_present_files(http_server: tuple[str, _ServerState], tmp_path: Path) -> None:
    """Verify a file already on disk is not re-fetched, and its sidecar is ensured."""
    base_url, state = http_server
    state.payload = b"unused"
    folder = tmp_path / "miscellaneous"
    folder.mkdir()
    (folder / "a.bin").write_bytes(b"already here")
    record = _record([DownloadRecord(file_name="a.bin", file_url=f"{base_url}/a.bin")])

    assert download_record_files(record, tmp_path) is True
    assert state.get_count == 0
    assert (folder / "a.bin").with_suffix(".sha256").is_file()


_ORIGIN_PAYLOAD = b"the genuine model weights payload" * 64


def _gateway_record(base_url: str, sha256: str) -> GenericModelRecord:
    """Return a one-file miscellaneous record whose origin is *base_url* and declared hash is *sha256*."""
    return _record([DownloadRecord(file_name="a.bin", file_url=f"{base_url}/a.bin", sha256sum=sha256)])


def test_gateway_is_preferred_and_apikey_forwarded(
    gateway_and_origin: tuple[tuple[str, _ServerState], tuple[str, _ServerState]], tmp_path: Path
) -> None:
    """Verify the gated mirror is used first (origin untouched) and the apikey header is forwarded to it."""
    (gateway_url, gateway_state), (origin_url, origin_state) = gateway_and_origin
    gateway_state.payload = _ORIGIN_PAYLOAD
    origin_state.payload = _ORIGIN_PAYLOAD
    record = _gateway_record(origin_url, _sha256(_ORIGIN_PAYLOAD))

    ok = download_record_files(record, tmp_path, gateway_base_url=gateway_url, apikey="k" * 22)

    assert ok is True
    assert (tmp_path / "miscellaneous" / "a.bin").read_bytes() == _ORIGIN_PAYLOAD
    assert gateway_state.get_count >= 1
    assert origin_state.get_count == 0
    assert "k" * 22 in gateway_state.seen_apikeys


@pytest.mark.parametrize("status", [401, 403, 404, 410])
def test_gateway_rejection_falls_back_to_origin(
    gateway_and_origin: tuple[tuple[str, _ServerState], tuple[str, _ServerState]],
    tmp_path: Path,
    status: int,
) -> None:
    """Verify an ineligible/absent gateway object (401/403/404/410) falls back transparently to the origin."""
    (gateway_url, gateway_state), (origin_url, origin_state) = gateway_and_origin
    gateway_state.force_status = status
    origin_state.payload = _ORIGIN_PAYLOAD
    record = _gateway_record(origin_url, _sha256(_ORIGIN_PAYLOAD))

    ok = download_record_files(record, tmp_path, gateway_base_url=gateway_url, apikey="k" * 22)

    assert ok is True
    assert (tmp_path / "miscellaneous" / "a.bin").read_bytes() == _ORIGIN_PAYLOAD
    assert gateway_state.get_count == 1  # definitive rejection: tried once, not retried
    assert origin_state.get_count >= 1


def test_gateway_corrupt_object_is_rejected_then_origin_used(
    gateway_and_origin: tuple[tuple[str, _ServerState], tuple[str, _ServerState]], tmp_path: Path
) -> None:
    """Verify a mirror object whose bytes fail the sha256 is discarded and the origin serves the real file."""
    (gateway_url, gateway_state), (origin_url, origin_state) = gateway_and_origin
    gateway_state.payload = b"tampered or stale bytes"
    origin_state.payload = _ORIGIN_PAYLOAD
    record = _gateway_record(origin_url, _sha256(_ORIGIN_PAYLOAD))

    ok = download_record_files(record, tmp_path, gateway_base_url=gateway_url, apikey="k" * 22)

    assert ok is True
    landed = tmp_path / "miscellaneous" / "a.bin"
    assert landed.read_bytes() == _ORIGIN_PAYLOAD
    assert not landed.with_suffix(".bin.part").exists()
    assert gateway_state.get_count >= 1
    assert origin_state.get_count >= 1


@pytest.mark.parametrize(
    ("url", "accepts"),
    [
        ("https://mirror.example", True),
        ("https://mirror.example/", True),
        ("http://localhost:8787", True),
        ("http://127.0.0.1:1234", True),
        ("http://mirror.example", False),  # plaintext non-local: would leak the key
        ("ftp://mirror.example", False),
        ("", False),
    ],
)
def test_gateway_accepts_key_requires_https_or_localhost(url: str, accepts: bool) -> None:
    """Verify the key is only deemed safe to send to an https gateway (or a localhost dev endpoint)."""
    assert gateway_accepts_key(url) is accepts


def test_insecure_http_gateway_is_skipped_and_key_never_sent(
    gateway_and_origin: tuple[tuple[str, _ServerState], tuple[str, _ServerState]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a plaintext gateway is never contacted (the apikey is not sent) and the origin serves the file.

    The local test servers run on 127.0.0.1, which is treated as a safe dev endpoint; emptying that set makes the
    http gateway count as insecure, exactly as a real ``http://`` non-local gateway would.
    """
    monkeypatch.setattr(download_engine, "_KEY_SAFE_HTTP_HOSTS", frozenset())
    (gateway_url, gateway_state), (origin_url, origin_state) = gateway_and_origin
    gateway_state.payload = _ORIGIN_PAYLOAD  # would serve if it were ever contacted
    origin_state.payload = _ORIGIN_PAYLOAD
    record = _gateway_record(origin_url, _sha256(_ORIGIN_PAYLOAD))

    ok = download_record_files(record, tmp_path, gateway_base_url=gateway_url, apikey="k" * 22)

    assert ok is True
    assert gateway_state.get_count == 0  # insecure gateway never contacted: the key was not sent in the clear
    assert origin_state.get_count >= 1


def test_gateway_skipped_without_apikey(
    gateway_and_origin: tuple[tuple[str, _ServerState], tuple[str, _ServerState]], tmp_path: Path
) -> None:
    """Verify the mirror path is inert without an apikey: the gateway is never contacted."""
    (gateway_url, gateway_state), (origin_url, origin_state) = gateway_and_origin
    gateway_state.force_status = 500  # would fail the request if it were ever attempted
    origin_state.payload = _ORIGIN_PAYLOAD
    record = _gateway_record(origin_url, _sha256(_ORIGIN_PAYLOAD))

    ok = download_record_files(record, tmp_path, gateway_base_url=gateway_url, apikey=None)

    assert ok is True
    assert gateway_state.get_count == 0
    assert origin_state.get_count >= 1


def test_configured_gateway_url_is_used_when_apikey_is_supplied(
    gateway_and_origin: tuple[tuple[str, _ServerState], tuple[str, _ServerState]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the settings-level gateway URL enables mirror-first downloads when the caller supplies an apikey."""
    (gateway_url, gateway_state), (origin_url, origin_state) = gateway_and_origin
    monkeypatch.setenv("HORDE_MODEL_REFERENCE_R2__GATEWAY_URL", gateway_url)
    gateway_state.payload = _ORIGIN_PAYLOAD
    origin_state.payload = _ORIGIN_PAYLOAD
    record = _gateway_record(origin_url, _sha256(_ORIGIN_PAYLOAD))

    ok = download_record_files(record, tmp_path, apikey="k" * 22)

    assert ok is True
    assert gateway_state.get_count >= 1
    assert origin_state.get_count == 0


def test_configured_gateway_can_be_disabled_per_call(
    gateway_and_origin: tuple[tuple[str, _ServerState], tuple[str, _ServerState]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify callers can force origin-only downloads even when the environment configures a gateway."""
    (gateway_url, gateway_state), (origin_url, origin_state) = gateway_and_origin
    monkeypatch.setenv("HORDE_MODEL_REFERENCE_R2__GATEWAY_URL", gateway_url)
    gateway_state.force_status = 500
    origin_state.payload = _ORIGIN_PAYLOAD
    record = _gateway_record(origin_url, _sha256(_ORIGIN_PAYLOAD))

    ok = download_record_files(record, tmp_path, apikey="k" * 22, use_configured_gateway=False)

    assert ok is True
    assert gateway_state.get_count == 0
    assert origin_state.get_count >= 1


def test_gateway_skipped_for_unhashed_record(
    gateway_and_origin: tuple[tuple[str, _ServerState], tuple[str, _ServerState]], tmp_path: Path
) -> None:
    """Verify a record with no real sha256 ('FIXME') cannot be content-addressed, so the gateway is skipped."""
    (gateway_url, gateway_state), (origin_url, origin_state) = gateway_and_origin
    gateway_state.force_status = 500
    origin_state.payload = _ORIGIN_PAYLOAD
    record = _gateway_record(origin_url, "FIXME")

    ok = download_record_files(record, tmp_path, gateway_base_url=gateway_url, apikey="k" * 22)

    assert ok is True
    assert gateway_state.get_count == 0
    assert origin_state.get_count >= 1


def test_gateway_5xx_falls_back_quickly_without_burning_retries(
    gateway_and_origin: tuple[tuple[str, _ServerState], tuple[str, _ServerState]], tmp_path: Path
) -> None:
    """Verify a 5xx gateway (a transient status, not a definitive rejection) hands off after a single attempt.

    A degraded mirror must not retry five times with backoff before reaching the origin: that would make the
    mirror slower than no mirror at all. The non-final candidate gets one attempt; the origin keeps the budget.
    """
    (gateway_url, gateway_state), (origin_url, origin_state) = gateway_and_origin
    gateway_state.force_status = 503
    origin_state.payload = _ORIGIN_PAYLOAD
    record = _gateway_record(origin_url, _sha256(_ORIGIN_PAYLOAD))

    ok = download_record_files(record, tmp_path, gateway_base_url=gateway_url, apikey="k" * 22)

    assert ok is True
    assert (tmp_path / "miscellaneous" / "a.bin").read_bytes() == _ORIGIN_PAYLOAD
    assert gateway_state.get_count == 1  # one attempt only, despite 503 not being a definitive rejection
    assert origin_state.get_count >= 1


def test_resume_probe_forwards_apikey_on_416(
    gateway_and_origin: tuple[tuple[str, _ServerState], tuple[str, _ServerState]], tmp_path: Path
) -> None:
    """Verify a complete .part is finalized from the gateway on 416 because the completeness probe is authed.

    The 416 path issues a second GET to read Content-Length; without forwarding the apikey the gated mirror
    would answer 401, the probe would read no length, and a complete partial would be wrongly discarded.
    """
    (gateway_url, gateway_state), (_origin_url, _origin_state) = gateway_and_origin
    gateway_state.payload = _ORIGIN_PAYLOAD
    destination = tmp_path / "miscellaneous" / "a.bin"
    destination.parent.mkdir(parents=True)
    (tmp_path / "miscellaneous" / "a.bin.part").write_bytes(_ORIGIN_PAYLOAD)  # already complete
    record = _gateway_record("http://127.0.0.1:1/unused", _sha256(_ORIGIN_PAYLOAD))

    ok = download_record_files(record, tmp_path, gateway_base_url=gateway_url, apikey="k" * 22)

    assert ok is True
    assert destination.read_bytes() == _ORIGIN_PAYLOAD
    assert all(seen == "k" * 22 for seen in gateway_state.seen_apikeys if seen != "")


def test_last_candidate_corrupt_file_is_removed(http_server: tuple[str, _ServerState], tmp_path: Path) -> None:
    """Verify a corrupt download from the only/last source leaves no file behind to be trusted on a later run."""
    base_url, state = http_server
    state.payload = b"these bytes do not match the declared hash"
    record = _record([DownloadRecord(file_name="a.bin", file_url=f"{base_url}/a.bin", sha256sum=_sha256(b"real"))])

    assert download_record_files(record, tmp_path) is False
    landed = tmp_path / "miscellaneous" / "a.bin"
    assert not landed.exists()
    assert not landed.with_suffix(".bin.part").exists()


def test_on_disk_file_failing_declared_hash_is_revalidated(
    http_server: tuple[str, _ServerState], tmp_path: Path
) -> None:
    """Verify an existing file whose bytes no longer match the declared hash is discarded and re-fetched."""
    base_url, state = http_server
    state.payload = b"the correct weights"
    folder = tmp_path / "miscellaneous"
    folder.mkdir()
    (folder / "a.bin").write_bytes(b"corrupt leftover from a past failure")
    record = _record(
        [DownloadRecord(file_name="a.bin", file_url=f"{base_url}/a.bin", sha256sum=_sha256(state.payload))],
    )

    assert download_record_files(record, tmp_path) is True
    assert (folder / "a.bin").read_bytes() == state.payload
    assert state.get_count >= 1


def test_download_file_does_not_retry_definitive_rejection(
    http_server: tuple[str, _ServerState], tmp_path: Path
) -> None:
    """Verify a 403 returns failure immediately without consuming the retry budget."""
    base_url, state = http_server
    state.force_status = 403
    destination = tmp_path / "model.bin"

    outcome = download_file(f"{base_url}/model.bin", destination, expected_sha256="deadbeef", max_retries=5)

    assert outcome.success is False
    assert state.get_count == 1


def test_engine_modules_import_without_torch() -> None:
    """Verify the layout and download engine are importable without torch or ComfyUI present."""
    import horde_model_reference.download_engine
    import horde_model_reference.on_disk_layout  # noqa: F401

    assert "torch" not in sys.modules
    assert "comfy" not in sys.modules
