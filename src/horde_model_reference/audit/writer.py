from __future__ import annotations

import json
import re
from pathlib import Path
from threading import RLock

from loguru import logger

from horde_model_reference.audit.events import AuditDomain, AuditEvent, AuditOperation, AuditPayload

DEFAULT_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024

_AUDIT_FILENAME_PATTERN = re.compile(r"audit-(\d{6})\.jsonl")


class AuditTrailWriter:
    """Append-only audit writer with size-based log rotation."""

    def __init__(self, *, root_path: Path, max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES) -> None:
        """Initialize the writer with a root directory and rotation threshold."""
        self._root_path = root_path
        self._root_path.mkdir(parents=True, exist_ok=True)
        self._max_file_size_bytes = max_file_size_bytes
        self._lock = RLock()
        self._state_path = self._root_path / "index.json"
        self._last_event_id = self._load_last_event_id()

    def append_event(
        self,
        *,
        domain: AuditDomain,
        category: str,
        model_name: str,
        operation: AuditOperation,
        logical_user_id: str,
        payload: AuditPayload | None = None,
        request_id: str | None = None,
        timestamp: int | None = None,
    ) -> AuditEvent:
        """Append a new audit event, returning the persisted object."""
        with self._lock:
            event_id = self._allocate_event_id()
            event = AuditEvent.new(
                event_id=event_id,
                domain=domain,
                category=category,
                model_name=model_name,
                operation=operation,
                logical_user_id=logical_user_id,
                payload=payload,
                request_id=request_id,
                timestamp=timestamp,
            )
            segment_path = self._resolve_segment_path(domain=domain, category=category)
            self._write_line(segment_path, event)
            return event

    def _allocate_event_id(self) -> int:
        self._last_event_id += 1
        self._state_path.write_text(json.dumps({"last_event_id": self._last_event_id}))
        return self._last_event_id

    def _load_last_event_id(self) -> int:
        if not self._state_path.exists():
            return 0
        try:
            data = json.loads(self._state_path.read_text() or "{}")
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            logger.warning(f"Unable to parse audit index file {self._state_path}: {exc}")
            return 0
        return int(data.get("last_event_id", 0))

    def _resolve_segment_path(self, *, domain: AuditDomain, category: str) -> Path:
        category_dir = self._root_path / domain.value / category
        category_dir.mkdir(parents=True, exist_ok=True)
        segments = sorted(category_dir.glob("audit-*.jsonl"))
        if not segments:
            return category_dir / "audit-000001.jsonl"
        latest = segments[-1]
        if latest.stat().st_size >= self._max_file_size_bytes:
            next_index = _extract_segment_index(latest) + 1
            return category_dir / f"audit-{next_index:06d}.jsonl"
        return latest

    def _write_line(self, path: Path, event: AuditEvent) -> None:
        serialized = json.dumps(
            event.model_dump(mode="json", exclude_none=True),
            separators=(",", ":"),
            ensure_ascii=True,
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")


def _extract_segment_index(path: Path) -> int:
    match = _AUDIT_FILENAME_PATTERN.match(path.name)
    if not match:
        return 1
    return int(match.group(1))
