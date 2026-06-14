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
from dataclasses import dataclass
from pathlib import Path

import pytest

from horde_model_reference.download_engine import (
    download_file,
    download_record_files,
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


@pytest.fixture
def http_server() -> Iterator[tuple[str, _ServerState]]:
    """Yield ``(base_url, state)`` for a running local HTTP server, shut down on teardown."""
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


def test_engine_modules_import_without_torch() -> None:
    """Verify the layout and download engine are importable without torch or ComfyUI present."""
    import horde_model_reference.download_engine
    import horde_model_reference.on_disk_layout  # noqa: F401

    assert "torch" not in sys.modules
    assert "comfy" not in sys.modules
