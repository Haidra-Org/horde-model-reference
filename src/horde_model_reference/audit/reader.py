from __future__ import annotations

from collections.abc import Collection, Iterator
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from horde_model_reference.audit.events import AuditDomain, AuditEvent


def _iter_dirs(path: Path) -> list[Path]:
    return [child for child in sorted(path.iterdir()) if child.is_dir()]


class AuditTrailReader:
    """Stream audit events from JSONL segments with optional filtering."""

    def __init__(self, *, root_path: Path) -> None:
        """Initialize the reader with the audit root directory."""
        self._root_path = Path(root_path)

    def iter_events(
        self,
        *,
        domains: Collection[AuditDomain] | None = None,
        categories: Collection[str] | None = None,
        model_names: Collection[str] | None = None,
        min_event_id: int | None = None,
        max_event_id: int | None = None,
        min_timestamp: int | None = None,
        max_timestamp: int | None = None,
    ) -> Iterator[AuditEvent]:
        """Yield AuditEvent objects matching the provided filters."""
        if not self._root_path.exists():
            return

        domain_filter = set(domains) if domains else None
        category_filter = set(categories) if categories else None
        model_filter = set(model_names) if model_names else None

        for domain_path in _iter_dirs(self._root_path):
            try:
                domain = AuditDomain(domain_path.name)
            except ValueError:
                logger.debug(f"Skipping unknown audit domain directory: {domain_path}")
                continue

            if domain_filter and domain not in domain_filter:
                continue

            for category_path in _iter_dirs(domain_path):
                category = category_path.name
                if category_filter and category not in category_filter:
                    continue

                for segment_path in sorted(category_path.glob("audit-*.jsonl")):
                    yield from self._iter_segment(
                        segment_path,
                        domain=domain,
                        category=category,
                        model_filter=model_filter,
                        min_event_id=min_event_id,
                        max_event_id=max_event_id,
                        min_timestamp=min_timestamp,
                        max_timestamp=max_timestamp,
                    )

    def _iter_segment(
        self,
        segment_path: Path,
        *,
        domain: AuditDomain,
        category: str,
        model_filter: set[str] | None,
        min_event_id: int | None,
        max_event_id: int | None,
        min_timestamp: int | None,
        max_timestamp: int | None,
    ) -> Iterator[AuditEvent]:
        try:
            with segment_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = AuditEvent.model_validate_json(line)
                    except (ValidationError, ValueError) as exc:
                        logger.warning(f"Skipping malformed audit event in {segment_path}: {exc}")
                        continue

                    if event.domain != domain or event.category != category:
                        continue

                    if model_filter and event.model_name not in model_filter:
                        continue

                    if min_event_id is not None and event.event_id < min_event_id:
                        continue

                    if max_event_id is not None and event.event_id > max_event_id:
                        continue

                    if min_timestamp is not None and event.timestamp < min_timestamp:
                        continue

                    if max_timestamp is not None and event.timestamp > max_timestamp:
                        continue

                    yield event
        except FileNotFoundError:
            logger.warning(f"Audit segment disappeared during iteration: {segment_path}")
        except OSError as exc:
            logger.warning(f"Unable to read audit segment {segment_path}: {exc}")
