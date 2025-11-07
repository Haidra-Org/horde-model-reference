"""FileSystem backend for PRIMARY mode.

This backend reads and writes model reference JSON files directly on the local filesystem.
It is the source of truth for PRIMARY mode instances and never interacts with GitHub.
"""

from __future__ import annotations

import contextlib
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, cast

import aiofiles
import httpx
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import override

from horde_model_reference import ReplicateMode, horde_model_reference_paths, horde_model_reference_settings
from horde_model_reference.backends.replica_backend_base import ReplicaBackendBase
from horde_model_reference.legacy.text_csv_utils import parse_legacy_text_csv
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_metadata import CategoryMetadata, MetadataManager, OperationType


class CategoryMetadataPopulationResult(BaseModel):
    """Result from ensure_category_metadata_populated method."""

    category_metadata_initialized: bool = Field(description="Whether v2 CategoryMetadata was initialized")
    legacy_metadata_initialized: bool = Field(description="Whether legacy CategoryMetadata was initialized")
    models_updated: int = Field(description="Number of models that had metadata populated")
    timestamp_used: int = Field(description="Unix timestamp used for metadata population")


class AllMetadataPopulationResult(BaseModel):
    """Result from ensure_all_metadata_populated method."""

    categories_processed: list[str] = Field(
        description="List of category names that were processed",
        default_factory=list,
    )
    total_categories: int = Field(
        description="Total number of categories processed",
        default=0,
    )
    total_models_updated: int = Field(
        description="Total number of models updated across all categories",
        default=0,
    )
    total_metadata_initialized: int = Field(
        description="Total number of metadata files initialized",
        default=0,
    )


class FileSystemBackend(ReplicaBackendBase):
    """Backend that reads/writes model references directly on the local filesystem."""

    def __init__(
        self,
        *,
        base_path: str | Path = horde_model_reference_paths.base_path,
        cache_ttl_seconds: int = 60,
        replicate_mode: ReplicateMode = ReplicateMode.PRIMARY,
        skip_startup_metadata_population: bool = False,
    ) -> None:
        """Initialize the FileSystem backend.

        Args:
            base_path: Base path for model reference files.
            cache_ttl_seconds: TTL for internal cache in seconds.
            replicate_mode: Must be PRIMARY.
            skip_startup_metadata_population: If True, skip automatic metadata population on startup.
                This is used when GitHub seeding will handle metadata population instead.

        Raises:
            ValueError: If replicate_mode is not PRIMARY.
        """
        if replicate_mode != ReplicateMode.PRIMARY:
            raise ValueError(
                "FileSystemBackend can only be used in PRIMARY mode. "
                "For REPLICA mode, use GitHubBackend or HTTPBackend."
            )
        super().__init__(mode=replicate_mode, cache_ttl_seconds=cache_ttl_seconds)

        self.base_path = Path(base_path)
        self._metadata_manager = MetadataManager(self.base_path)

        logger.debug(f"FileSystemBackend initialized with base_path={self.base_path}")

        # Create empty files for categories that have no legacy format available
        # This ensures consistent behavior between CI (fresh environment) and local (may have existing files)
        from horde_model_reference.meta_consts import no_legacy_format_available_categories

        for category in no_legacy_format_available_categories:
            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )
            if file_path and not file_path.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("{}")
                logger.info(f"Created empty file for {category} (no legacy format available)")

        # Populate metadata on startup if not skipped
        if not skip_startup_metadata_population:
            logger.info("Running startup metadata population check")
            self.ensure_all_metadata_populated()
        else:
            logger.debug("Skipping startup metadata population (will be handled by GitHub seeding)")

    def _resolve_legacy_text_generation_path(self) -> tuple[Path, bool]:
        """Return the legacy text_generation path and whether it is CSV-based."""
        csv_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
            MODEL_REFERENCE_CATEGORY.text_generation,
            base_path=self.base_path,
        )
        json_path = csv_path.with_name("text_generation.json")

        if horde_model_reference_settings.canonical_format == "legacy" and (
            json_path.exists() or not csv_path.exists()
        ):
            return json_path, False

        if csv_path.exists():
            return csv_path, True

        if json_path.exists():
            return json_path, False

        if horde_model_reference_settings.canonical_format == "legacy":
            return json_path, False

        return csv_path, True

    @override
    def _get_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Return the file path for mtime validation.

        Args:
            category: The category to get the file path for.

        Returns:
            Path | None: Path to file for mtime validation.
        """
        return horde_model_reference_paths.get_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

    @override
    def _get_legacy_file_path_for_validation(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Return the legacy file path for mtime validation.

        Args:
            category: The category to get the legacy file path for.

        Returns:
            Path | None: Path to legacy file for mtime validation.
        """
        return horde_model_reference_paths.get_legacy_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

    def _mark_category_modified(self, category: MODEL_REFERENCE_CATEGORY, file_path: Path) -> None:
        """Mark a category as modified after a write operation.

        This invalidates the cache and triggers callbacks to notify manager.

        Args:
            category: Category that was modified.
            file_path: Path to the file that was modified.
        """
        # Use mark_stale() to trigger callbacks, not _invalidate_cache()
        self.mark_stale(category)
        logger.debug(f"Marked category {category} as modified")

    def _mark_legacy_category_modified(self, category: MODEL_REFERENCE_CATEGORY, legacy_file_path: Path) -> None:
        """Mark a legacy category as modified after a write operation.

        This invalidates the legacy cache and triggers callbacks.

        Args:
            category: Category that was modified.
            legacy_file_path: Path to the legacy file that was modified.
        """
        # Use mark_stale() to trigger callbacks
        self.mark_stale(category)
        logger.debug(f"Marked legacy category {category} as modified")

    def _read_legacy_csv_to_dict(self, file_path: Path) -> dict[str, Any]:
        """Read legacy CSV file (models.csv format) and convert to dict format.

        This reads the legacy models.csv format and converts it to the
        grouped dict format (one entry per base model, no backend prefixes).
        Follows the same logic as scripts/legacy_text/convert.py.

        Args:
            file_path: Path to the legacy CSV file.

        Returns:
            dict[str, Any]: Model data with one entry per base model.

        Raises:
            Exception: If CSV parsing fails.
        """
        data: dict[str, Any] = {}
        parsed_rows, parse_issues = parse_legacy_text_csv(file_path)
        for issue in parse_issues:
            logger.warning(f"Legacy CSV issue for {issue.row_identifier}: {issue.message}")

        for csv_row in parsed_rows:
            name = csv_row.name
            model_name = name.split("/")[1] if "/" in name else name
            tags = set(csv_row.tags)
            if csv_row.style:
                tags.add(csv_row.style)
            tags.add(f"{round(csv_row.parameters_bn, 0):.0f}B")

            settings_dict = dict(csv_row.settings) if csv_row.settings is not None else {}

            display_name = csv_row.display_name
            if not display_name:
                display_name = re.sub(r" +", " ", re.sub(r"[-_]", " ", model_name)).strip()

            record: dict[str, Any] = {
                "name": name,
                "model_name": model_name,
                "parameters": csv_row.parameters,
                "description": csv_row.description,
                "version": csv_row.version,
                "style": csv_row.style,
                "nsfw": csv_row.nsfw,
                "baseline": csv_row.baseline,
                "url": csv_row.url,
                "tags": sorted(tags),
                "settings": settings_dict,
                "display_name": display_name,
            }

            record = {k: v for k, v in record.items() if v or v is False}
            data[name] = record

        logger.debug(f"Read {len(data)} models from legacy CSV (grouped, no backend prefixes) from {file_path}")
        return data

    def _read_csv_to_dict(self, file_path: Path) -> dict[str, Any]:
        """Read CSV file and convert to dict format (grouped by base name, no backend prefixes).

        This reads the grouped CSV format and returns a dict with one entry per base model.
        No backend prefix duplicates are generated here - that only happens during GitHub sync.

        Args:
            file_path: Path to the CSV file.

        Returns:
            dict[str, Any]: Model data with one entry per base model.

        Raises:
            Exception: If CSV parsing fails.
        """
        data: dict[str, Any] = {}
        parsed_rows, parse_issues = parse_legacy_text_csv(file_path)
        for issue in parse_issues:
            logger.warning(f"CSV issue for {issue.row_identifier}: {issue.message}")

        for csv_row in parsed_rows:
            name = csv_row.name
            model_name = name.split("/")[1] if "/" in name else name
            tags = set(csv_row.tags)
            if csv_row.style:
                tags.add(csv_row.style)
            tags.add(f"{round(csv_row.parameters_bn, 0):.0f}B")

            settings_dict = dict(csv_row.settings) if csv_row.settings is not None else {}

            display_name = csv_row.display_name
            if not display_name:
                display_name = re.sub(r" +", " ", re.sub(r"[-_]", " ", model_name)).strip()

            record: dict[str, Any] = {
                "name": name,
                "model_name": model_name,
                "parameters": csv_row.parameters,
                "description": csv_row.description,
                "version": csv_row.version,
                "style": csv_row.style,
                "nsfw": csv_row.nsfw,
                "baseline": csv_row.baseline,
                "url": csv_row.url,
                "tags": sorted(tags),
                "settings": settings_dict,
                "display_name": display_name,
            }

            record = {k: v for k, v in record.items() if v or v is False}
            data[name] = record

        logger.debug(f"Read {len(data)} models from CSV (grouped, no backend prefixes) from {file_path}")
        return data

    def _write_dict_to_csv(self, data: dict[str, Any], file_path: Path) -> None:
        """Write dict format to CSV file (removes backend prefix duplicates).

        This writes the grouped CSV format with one entry per base model.
        Any backend-prefixed entries in the input are filtered out.

        Args:
            data: Model data dict (may contain backend-prefixed duplicates).
            file_path: Path to write the CSV file.

        Raises:
            Exception: If CSV writing fails.
        """
        from horde_model_reference.meta_consts import has_legacy_text_backend_prefix

        # Filter out backend-prefixed entries
        base_models: dict[str, Any] = {}
        for model_name, record in data.items():
            if has_legacy_text_backend_prefix(model_name):
                logger.debug(f"Skipping backend-prefixed entry during CSV write: {model_name}")
                continue
            base_models[model_name] = record

        # Convert to CSV rows
        csv_rows: list[dict[str, str]] = []

        for model_name, record in base_models.items():
            # Extract parameters in billions
            parameters_int = record.get("parameters", 0)
            params_bn = float(parameters_int) / 1_000_000_000

            # Extract tags and remove auto-generated ones
            tags = record.get("tags", [])
            tags_set = set(tags) if isinstance(tags, list) else set()

            # Remove style tag
            style = record.get("style")
            if style and style in tags_set:
                tags_set.discard(style)

            # Remove parameter size tag
            size_tag = f"{round(params_bn, 0):.0f}B"
            tags_set.discard(size_tag)

            # Serialize settings to compact JSON
            settings = record.get("settings", {})
            settings_str = json.dumps(settings, separators=(",", ":")) if settings else ""

            # Check if display_name is auto-generated (if so, omit it)
            display_name = record.get("display_name", "")
            model_name_field = record.get("model_name", model_name)
            auto_display = re.sub(r" +", " ", re.sub(r"[-_]", " ", model_name_field)).strip()
            if display_name == auto_display:
                display_name = ""

            csv_row = {
                "name": model_name,
                "parameters_bn": f"{params_bn:.1f}",
                "description": record.get("description", ""),
                "version": record.get("version", ""),
                "style": style or "",
                "nsfw": str(record.get("nsfw", False)).lower(),
                "baseline": record.get("baseline", ""),
                "url": record.get("url", ""),
                "tags": ",".join(sorted(tags_set)),
                "settings": settings_str,
                "display_name": display_name,
            }

            csv_rows.append(csv_row)

        # Write CSV file
        fieldnames = [
            "name",
            "parameters_bn",
            "description",
            "version",
            "style",
            "nsfw",
            "baseline",
            "url",
            "tags",
            "settings",
            "display_name",
        ]

        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        logger.debug(f"Wrote {len(csv_rows)} models to CSV (grouped, no backend prefixes) to {file_path}")

    @override
    def fetch_category(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch model reference data for a specific category from filesystem.

        All v2 format files (including text_generation.json) are in JSON format.
        CSV format is only used for legacy files (legacy/models.csv).

        Args:
            category: The category to fetch.
            force_refresh: If True, bypass cache and read from disk.

        Returns:
            dict[str, Any] | None: The model reference data, or None if file doesn't exist.
        """
        with self._lock:
            if not (force_refresh or self.should_fetch_data(category)):
                return self._get_from_cache(category)

            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if not file_path or not file_path.exists():
                logger.debug(f"File not found for {category}: {file_path}")
                self._store_in_cache(category, None)
                return None

            try:
                # All v2 files are JSON format (including text_generation.json)
                with open(file_path, encoding="utf-8") as f:
                    data: dict[str, Any] = json.load(f)

                self._store_in_cache(category, data)
                logger.debug(f"Loaded {category} from {file_path}")
                return data

            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                self._invalidate_cache(category)
                return None

    @override
    def fetch_all_categories(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Fetch model reference data for all categories.

        Args:
            force_refresh: If True, bypass cache for all categories.

        Returns:
            dict mapping categories to their model reference data.
        """
        result: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}

        for category in MODEL_REFERENCE_CATEGORY:
            result[category] = self.fetch_category(category, force_refresh=force_refresh)

        return result

    async def fetch_category_async(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any] | None:
        """Asynchronously fetch model reference data for a category.

        Note: File I/O is still synchronous as async file I/O doesn't provide
        significant benefits for small JSON files.

        Args:
            category: The category to fetch.
            httpx_client: Optional httpx async client for downloads.
            force_refresh: If True, bypass cache.

        Returns:
            dict[str, Any] | None: The model reference data.
        """
        async with self._async_lock:
            if not (force_refresh or self.should_fetch_data(category)):
                return self._get_from_cache(category)

            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )
            if not file_path or not file_path.exists():
                logger.debug(f"File not found for {category}: {file_path}")
                self._store_in_cache(category, None)
                return None
            try:
                async with aiofiles.open(file_path, encoding="utf-8") as f:
                    content = await f.read()
                    data: dict[str, Any] = json.loads(content)

                self._store_in_cache(category, data)
                logger.debug(f"Loaded {category} from {file_path} asynchronously")
                return data
            except Exception as e:
                logger.error(f"Failed to read {file_path} asynchronously: {e}")
                self._invalidate_cache(category)
                return None

    @override
    async def fetch_all_categories_async(
        self,
        *,
        httpx_client: httpx.AsyncClient | None = None,
        force_refresh: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None]:
        """Asynchronously fetch all categories (delegates to sync method)."""
        result: dict[MODEL_REFERENCE_CATEGORY, dict[str, Any] | None] = {}

        for category in MODEL_REFERENCE_CATEGORY:
            result[category] = await self.fetch_category_async(
                category,
                httpx_client=httpx_client,
                force_refresh=force_refresh,
            )

        return result

    @override
    def get_category_file_path(self, category: MODEL_REFERENCE_CATEGORY) -> Path | None:
        """Get the file path for a category's data.

        Args:
            category: The category to get path for.

        Returns:
            Path | None: Path to the JSON file, or None if not configured.
        """
        return horde_model_reference_paths.get_model_reference_file_path(
            category,
            base_path=self.base_path,
        )

    @override
    def get_all_category_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Get file paths for all categories.

        Returns:
            dict: Mapping of categories to their file paths.
        """
        return horde_model_reference_paths.get_all_model_reference_file_paths(base_path=self.base_path)

    @override
    def get_legacy_json(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> dict[str, Any] | None:
        """Get legacy format data from legacy/ folder.

        For text_generation category, reads from CSV format (models.csv).
        For other categories, reads from JSON format.

        Args:
            category: Category to retrieve.
            redownload: If True, bypass cache and read from disk.

        Returns:
            dict[str, Any] | None: The legacy format data, or None if file doesn't exist.
        """
        with self._lock:
            # Check cache first unless redownload
            if not redownload:
                legacy_dict, _ = self._get_legacy_from_cache(category)
                if legacy_dict is not None:
                    return legacy_dict

            if category == MODEL_REFERENCE_CATEGORY.text_generation:
                legacy_file_path, is_csv = self._resolve_legacy_text_generation_path()
                target_write_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
                    category,
                    base_path=self.base_path,
                )
            else:
                legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
                    category,
                    base_path=self.base_path,
                )
                target_write_path = legacy_file_path
                is_csv = False

            if not legacy_file_path.exists():
                logger.debug(f"Legacy file not found for {category}: {legacy_file_path}")
                self._store_legacy_in_cache(category, None, None)
                return None

            try:
                # Handle CSV format for text_generation
                if category == MODEL_REFERENCE_CATEGORY.text_generation:
                    if is_csv:
                        data = self._read_legacy_csv_to_dict(legacy_file_path)
                    else:
                        with open(legacy_file_path, encoding="utf-8") as f:
                            data = cast(dict[str, Any], json.load(f))

                    # Generate JSON string for cache
                    content = json.dumps(data, indent=2, ensure_ascii=False)
                    self._store_legacy_in_cache(category, data, content)
                    logger.debug(f"Loaded legacy CSV for {category} from {legacy_file_path}")
                    return data

                # Handle JSON format for other categories
                with open(legacy_file_path, encoding="utf-8") as f:
                    content = f.read()
                    data = cast(dict[str, Any], json.loads(content))

                self._store_legacy_in_cache(category, data, content)
                logger.debug(f"Loaded legacy JSON for {category} from {legacy_file_path}")
                return data

            except Exception as e:
                logger.error(f"Failed to read legacy file {legacy_file_path}: {e}")
                self._invalidate_legacy_cache(category)
                return None

    @override
    def get_legacy_json_string(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        redownload: bool = False,
    ) -> str | None:
        """Get legacy format data as JSON string from legacy/ folder.

        For text_generation category, reads CSV and converts to JSON string.
        For other categories, reads JSON format directly.

        Args:
            category: Category to retrieve.
            redownload: If True, bypass cache and read from disk.

        Returns:
            str | None: The legacy format as JSON string, or None if file doesn't exist.
        """
        with self._lock:
            # Check cache first unless redownload
            if not redownload:
                _, legacy_string = self._get_legacy_from_cache(category)
                if legacy_string is not None:
                    return legacy_string

            if category == MODEL_REFERENCE_CATEGORY.text_generation:
                legacy_file_path, is_csv = self._resolve_legacy_text_generation_path()
            else:
                legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
                    category,
                    base_path=self.base_path,
                )
                is_csv = False

            if not legacy_file_path.exists():
                logger.debug(f"Legacy file not found for {category}: {legacy_file_path}")
                self._store_legacy_in_cache(category, None, None)
                return None

            try:
                # Handle CSV format for text_generation
                if category == MODEL_REFERENCE_CATEGORY.text_generation:
                    if is_csv:
                        data = self._read_legacy_csv_to_dict(legacy_file_path)
                    else:
                        with open(legacy_file_path, encoding="utf-8") as f:
                            data = cast(dict[str, Any], json.load(f))
                    # Generate JSON string
                    content = json.dumps(data, indent=2, ensure_ascii=False)
                    self._store_legacy_in_cache(category, data, content)
                    logger.debug(f"Loaded legacy CSV string for {category} from {legacy_file_path}")
                    return content

                # Handle JSON format for other categories
                with open(legacy_file_path, encoding="utf-8") as f:
                    content = f.read()
                    data = cast(dict[str, Any], json.loads(content))

                self._store_legacy_in_cache(category, data, content)
                logger.debug(f"Loaded legacy JSON string for {category} from {legacy_file_path}")
                return content

            except Exception as e:
                logger.error(f"Failed to read legacy file {legacy_file_path}: {e}")
                self._invalidate_legacy_cache(category)
                return None

    @override
    def supports_writes(self) -> bool:
        """Check if backend supports writes (always True for PRIMARY filesystem).

        Returns:
            bool: Always True.
        """
        return True

    @override
    def supports_metadata(self) -> bool:
        """Check if backend supports metadata tracking (always True for PRIMARY filesystem).

        Returns:
            bool: Always True.
        """
        return True

    @override
    def update_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record_dict: dict[str, Any],
    ) -> None:
        """Update or create a model reference.

        For text_generation category, modifies the CSV file on disk atomically.
        For other categories, modifies the JSON file on disk atomically.
        Preserves created_at and created_by metadata on updates.

        Args:
            category: The category to update.
            model_name: The name of the model to update or create.
            record_dict: The model record data as a dictionary.

        Raises:
            FileNotFoundError: If the category file path is not configured.
        """
        from horde_model_reference.meta_consts import has_legacy_text_backend_prefix

        with self._lock:
            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if not file_path:
                raise FileNotFoundError(f"No file path configured for category {category}")

            # Read existing data (CSV for text_generation, JSON for others)
            existing_data: dict[str, Any]
            if file_path.exists():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        existing_data = json.load(f)
                except json.JSONDecodeError:
                    if category == MODEL_REFERENCE_CATEGORY.text_generation:
                        existing_data = self._read_csv_to_dict(file_path)
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
                    raise
            else:
                existing_data = {}
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # For text_generation, filter out backend prefix entries before updating
            if category == MODEL_REFERENCE_CATEGORY.text_generation and has_legacy_text_backend_prefix(model_name):
                logger.warning(
                    f"Attempted to update backend-prefixed model {model_name} in text_generation. "
                    "Backend prefixes are not stored internally - update the base model instead."
                )
                # Don't raise an error, just skip the update
                return

            # Determine if this is a create or update operation
            is_update = model_name in existing_data
            operation_type = OperationType.UPDATE if is_update else OperationType.CREATE

            # Handle per-model metadata
            if is_update:
                # Preserve creation metadata if model already exists
                existing_model = existing_data[model_name]
                self._metadata_manager.model_metadata.preserve_creation_fields(existing_model, record_dict)
                # Set updated_at timestamp
                self._metadata_manager.model_metadata.set_update_timestamp(record_dict)
            else:
                # For new models, ensure timestamps are populated (without overwriting existing values)
                self._metadata_manager.model_metadata.ensure_metadata_populated(record_dict)

            existing_data[model_name] = record_dict

            temp_path = file_path.with_suffix(f".tmp.{time.time()}")
            try:
                # Write to temp file (CSV for text_generation, JSON for others)
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    try:
                        import os

                        os.fsync(f.fileno())
                    except Exception:
                        pass

                # Atomic replace
                if file_path.exists():
                    backup_path = file_path.with_suffix(".bak")
                    file_path.replace(backup_path)
                    temp_path.replace(file_path)
                    with contextlib.suppress(Exception):
                        backup_path.unlink()
                else:
                    temp_path.replace(file_path)

                logger.info(f"Updated model {model_name} in category {category} at {file_path}")

                # Record metadata for observability (centralized hook point)
                self._metadata_manager.record_v2_operation(
                    category=category,
                    operation=operation_type,
                    model_name=model_name,
                    success=True,
                    backend_type=self.__class__.__name__,
                )

                self._mark_category_modified(category, file_path)

            except Exception as e:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                logger.error(f"Failed to update model {model_name} in {category}: {e}")
                raise

    @override
    def delete_model(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> None:
        """Delete a model reference.

        For text_generation category, removes the model from the CSV file on disk atomically.
        For other categories, removes the model from the JSON file on disk atomically.

        Args:
            category: The category containing the model.
            model_name: The name of the model to delete.

        Raises:
            FileNotFoundError: If the category file doesn't exist.
            KeyError: If the model doesn't exist in the category.
        """
        with self._lock:
            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            existing_data: dict[str, Any]
            if not file_path or not file_path.exists():
                raise FileNotFoundError(f"Category file not found: {file_path}")

            # Read existing data (CSV for text_generation, JSON for others)
            try:
                with open(file_path, encoding="utf-8") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                if category == MODEL_REFERENCE_CATEGORY.text_generation:
                    existing_data = self._read_csv_to_dict(file_path)
                else:
                    raise
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                raise

            if model_name not in existing_data:
                raise KeyError(f"Model {model_name} not found in category {category}")

            del existing_data[model_name]

            temp_path = file_path.with_suffix(f".tmp.{time.time()}")
            try:
                # Write to temp file (CSV for text_generation, JSON for others)
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    try:
                        import os

                        os.fsync(f.fileno())
                    except Exception:
                        pass

                backup_path = file_path.with_suffix(".bak")
                file_path.replace(backup_path)
                temp_path.replace(file_path)

                with contextlib.suppress(Exception):
                    backup_path.unlink()

                logger.info(f"Deleted model {model_name} from category {category} at {file_path}")

                # Record metadata for observability (centralized hook point)
                self._metadata_manager.record_v2_operation(
                    category=category,
                    operation=OperationType.DELETE,
                    model_name=model_name,
                    success=True,
                    backend_type=self.__class__.__name__,
                )

                self._mark_category_modified(category, file_path)

            except Exception as e:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                logger.error(f"Failed to delete model {model_name} from {category}: {e}")
                raise

    @override
    def supports_legacy_writes(self) -> bool:
        """Check if backend supports legacy format writes.

        Returns True only when canonical_format='legacy' in settings.

        Returns:
            bool: True if legacy writes are supported.
        """
        from horde_model_reference import horde_model_reference_settings

        return horde_model_reference_settings.canonical_format == "legacy"

    @override
    def update_model_legacy(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
        record_dict: dict[str, Any],
    ) -> None:
        """Update or create a model reference in legacy format.

        This method modifies the legacy format JSON file on disk atomically.

        Args:
            category: The category to update.
            model_name: The name of the model to update or create.
            record_dict: The model record data in legacy format as a dictionary.

        Raises:
            FileNotFoundError: If the legacy category file path is not configured.
            RuntimeError: If canonical_format is not set to 'legacy'.
        """
        from horde_model_reference import horde_model_reference_settings

        if not self.supports_legacy_writes():
            raise RuntimeError(
                "Legacy writes are only supported when canonical_format='legacy'. "
                f"Current setting: canonical_format='{horde_model_reference_settings.canonical_format}'"
            )

        with self._lock:
            if category == MODEL_REFERENCE_CATEGORY.text_generation:
                legacy_file_path, is_csv = self._resolve_legacy_text_generation_path()
                target_write_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
                    category,
                    base_path=self.base_path,
                )
            else:
                legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
                    category,
                    base_path=self.base_path,
                )
                target_write_path = legacy_file_path
                is_csv = False

            if not legacy_file_path:
                raise FileNotFoundError(f"No legacy file path configured for category {category}")

            existing_data: dict[str, Any]
            if legacy_file_path.exists():
                try:
                    if category == MODEL_REFERENCE_CATEGORY.text_generation and is_csv:
                        # When canonical format was previously v2, the CSV may still exist.
                        # Promote it to JSON structure for legacy CRUD operations.
                        existing_data = self._read_legacy_csv_to_dict(legacy_file_path)
                    else:
                        with open(legacy_file_path, encoding="utf-8") as f:
                            existing_data = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to read {legacy_file_path}: {e}")
                    raise
            else:
                existing_data = {}
                target_write_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine if this is a create or update operation
            is_update = model_name in existing_data
            operation_type = OperationType.UPDATE if is_update else OperationType.CREATE

            existing_data[model_name] = record_dict

            temp_path = target_write_path.with_suffix(f".tmp.{time.time()}")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    try:
                        import os

                        os.fsync(f.fileno())
                    except Exception:
                        pass

                if target_write_path.exists():
                    backup_path = target_write_path.with_suffix(".bak")
                    target_write_path.replace(backup_path)
                    temp_path.replace(target_write_path)
                    with contextlib.suppress(Exception):
                        backup_path.unlink()
                else:
                    temp_path.replace(target_write_path)

                logger.info(f"Updated legacy model {model_name} in category {category} at {target_write_path}")

                # Record metadata for observability (centralized hook point)
                self._metadata_manager.record_legacy_operation(
                    category=category,
                    operation=operation_type,
                    model_name=model_name,
                    success=True,
                    backend_type=self.__class__.__name__,
                )

                self._mark_legacy_category_modified(category, target_write_path)

                if legacy_file_path != target_write_path and legacy_file_path.exists():
                    with contextlib.suppress(Exception):
                        legacy_file_path.unlink()

            except Exception as e:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                logger.error(f"Failed to update legacy model {model_name} in {category}: {e}")
                raise

    @override
    def delete_model_legacy(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        model_name: str,
    ) -> None:
        """Delete a model reference from legacy format files.

        This method removes the model from the legacy format JSON file on disk atomically.

        Args:
            category: The category containing the model.
            model_name: The name of the model to delete.

        Raises:
            FileNotFoundError: If the legacy category file doesn't exist.
            KeyError: If the model doesn't exist in the category.
            RuntimeError: If canonical_format is not set to 'legacy'.
        """
        from horde_model_reference import horde_model_reference_settings

        if not self.supports_legacy_writes():
            raise RuntimeError(
                "Legacy writes are only supported when canonical_format='legacy'. "
                f"Current setting: canonical_format='{horde_model_reference_settings.canonical_format}'"
            )

        with self._lock:
            if category == MODEL_REFERENCE_CATEGORY.text_generation:
                legacy_file_path, is_csv = self._resolve_legacy_text_generation_path()
                target_write_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
                    category,
                    base_path=self.base_path,
                )
            else:
                legacy_file_path = horde_model_reference_paths.get_legacy_model_reference_file_path(
                    category,
                    base_path=self.base_path,
                )
                target_write_path = legacy_file_path
                is_csv = False

            existing_data: dict[str, Any]
            if not legacy_file_path or not legacy_file_path.exists():
                raise FileNotFoundError(f"Legacy category file not found: {legacy_file_path}")

            try:
                if category == MODEL_REFERENCE_CATEGORY.text_generation and is_csv:
                    existing_data = self._read_legacy_csv_to_dict(legacy_file_path)
                else:
                    with open(legacy_file_path, encoding="utf-8") as f:
                        existing_data = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read {legacy_file_path}: {e}")
                raise

            if model_name not in existing_data:
                raise KeyError(f"Model {model_name} not found in legacy category {category}")

            del existing_data[model_name]

            temp_path = target_write_path.with_suffix(f".tmp.{time.time()}")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    try:
                        import os

                        os.fsync(f.fileno())
                    except Exception:
                        pass

                if target_write_path.exists():
                    backup_path = target_write_path.with_suffix(".bak")
                    target_write_path.replace(backup_path)
                    temp_path.replace(target_write_path)
                    with contextlib.suppress(Exception):
                        backup_path.unlink()
                else:
                    temp_path.replace(target_write_path)

                logger.info(f"Deleted legacy model {model_name} from category {category} at {target_write_path}")

                # Record metadata for observability (centralized hook point)
                self._metadata_manager.record_legacy_operation(
                    category=category,
                    operation=OperationType.DELETE,
                    model_name=model_name,
                    success=True,
                    backend_type=self.__class__.__name__,
                )

                self._mark_legacy_category_modified(category, target_write_path)

                if legacy_file_path != target_write_path and legacy_file_path.exists():
                    with contextlib.suppress(Exception):
                        legacy_file_path.unlink()

            except Exception as e:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                logger.error(f"Failed to delete legacy model {model_name} from {category}: {e}")
                raise

    def _populate_model_metadata(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        timestamp: int | None = None,
    ) -> int:
        """Populate missing per-model metadata fields in a category's JSON file.

        This method scans all models in a category file and ensures each has:
        - metadata.created_at (if missing)
        - metadata.updated_at (if missing)

        Args:
            category: The category to populate metadata for.
            timestamp: The timestamp to use for created_at/updated_at. If None, uses current time.

        Returns:
            int: Number of models that had metadata populated.
        """
        with self._lock:
            file_path = horde_model_reference_paths.get_model_reference_file_path(
                category,
                base_path=self.base_path,
            )

            if not file_path or not file_path.exists():
                logger.debug(f"Category file not found for {category}, skipping metadata population")
                return 0

            if timestamp is None:
                timestamp = int(time.time())

            try:
                with open(file_path, encoding="utf-8") as f:
                    data: dict[str, Any] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read {file_path} for metadata population: {e}")
                return 0

            models_updated = 0
            for _model_name, model_data in data.items():
                if not isinstance(model_data, dict):
                    continue

                # Use ModelMetadataManager to ensure metadata is populated
                if self._metadata_manager.model_metadata.ensure_metadata_populated(model_data, timestamp):
                    models_updated += 1

            if models_updated == 0:
                logger.trace(f"No models needed metadata population in {category}")
                return 0

            temp_path = file_path.with_suffix(f".tmp.{time.time()}")
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    try:
                        import os

                        os.fsync(f.fileno())
                    except Exception:
                        pass

                backup_path = file_path.with_suffix(".bak")
                file_path.replace(backup_path)
                temp_path.replace(file_path)

                with contextlib.suppress(Exception):
                    backup_path.unlink()

                logger.info(f"Populated metadata for {models_updated} models in {category}")
                self._mark_category_modified(category, file_path)

                return models_updated

            except Exception as e:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                logger.error(f"Failed to write metadata-populated file for {category}: {e}")
                return 0

    def ensure_category_metadata_populated(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        timestamp: int | None = None,
    ) -> CategoryMetadataPopulationResult:
        """Ensure both CategoryMetadata and per-model metadata are populated for a category.

        This method:
        1. Checks if CategoryMetadata exists for both v2 and legacy formats
        2. Initializes CategoryMetadata if missing
        3. Populates per-model metadata fields in JSON files
        4. Uses the same timestamp for both backend and model-level metadata

        Args:
            category: The category to ensure metadata for.
            timestamp: Optional timestamp to use. If None, uses current time.

        Returns:
            dict with keys:
                - "category_metadata_initialized": bool
                - "legacy_metadata_initialized": bool
                - "models_updated": int
                - "timestamp_used": int
        """
        with self._lock:
            if timestamp is None:
                timestamp = int(time.time())

            result = CategoryMetadataPopulationResult(
                category_metadata_initialized=False,
                legacy_metadata_initialized=False,
                models_updated=0,
                timestamp_used=timestamp,
            )

            # Get or initialize v2 CategoryMetadata
            v2_metadata = self._metadata_manager.get_or_initialize_v2_metadata(
                category=category,
                backend_type=self.__class__.__name__,
            )

            # Check if it was just created (no prior data file existed)
            if v2_metadata.initialization_time == v2_metadata.last_updated:
                result.category_metadata_initialized = True
                logger.trace(f"Initialized v2 CategoryMetadata for {category}")

            # Use initialization_time from metadata
            timestamp = v2_metadata.initialization_time
            result.timestamp_used = timestamp

            # Get or initialize legacy CategoryMetadata
            legacy_metadata = self._metadata_manager.get_or_initialize_legacy_metadata(
                category=category,
                backend_type=self.__class__.__name__,
            )

            # Check if it was just created
            if legacy_metadata.initialization_time == legacy_metadata.last_updated:
                result.legacy_metadata_initialized = True
                logger.trace(f"Initialized legacy CategoryMetadata for {category}")

            # Populate per-model metadata using the determined timestamp
            models_updated = self._populate_model_metadata(category, timestamp)
            result.models_updated = models_updated

            if result.category_metadata_initialized or result.legacy_metadata_initialized or models_updated > 0:
                logger.info(
                    f"Metadata population for {category}: "
                    f"v2_meta={result.category_metadata_initialized}, "
                    f"legacy_meta={result.legacy_metadata_initialized}, "
                    f"models={models_updated}"
                )

            return result

    def ensure_all_metadata_populated(self) -> AllMetadataPopulationResult:
        """Ensure metadata is populated for all categories that have files.

        Scans all category files and ensures:
        1. CategoryMetadata exists (both v2 and legacy formats)
        2. All model records have metadata fields populated

        This is called:
        - On FileSystemBackend initialization (PRIMARY mode)
        - After GitHub seeding completes

        Returns:
            AllMetadataPopulationResult with summary of metadata population.
        """
        with self._lock:
            result = AllMetadataPopulationResult()

            logger.info("Starting metadata population scan for all categories")

            for category in MODEL_REFERENCE_CATEGORY:
                file_path = horde_model_reference_paths.get_model_reference_file_path(
                    category,
                    base_path=self.base_path,
                )

                # Skip categories that don't have files
                if not file_path or not file_path.exists():
                    logger.debug(f"Skipping {category} - no file found")
                    continue

                # Ensure metadata for this category
                category_result = self.ensure_category_metadata_populated(category)

                if (
                    category_result.category_metadata_initialized
                    or category_result.legacy_metadata_initialized
                    or category_result.models_updated > 0
                ):
                    result.categories_processed.append(category.value)
                    result.total_categories += 1
                    result.total_models_updated += category_result.models_updated

                    if category_result.category_metadata_initialized:
                        result.total_metadata_initialized += 1
                    if category_result.legacy_metadata_initialized:
                        result.total_metadata_initialized += 1

            if result.total_categories > 0:
                logger.info(
                    f"Metadata population complete: "
                    f"{result.total_categories} categories processed, "
                    f"{result.total_models_updated} models updated, "
                    f"{result.total_metadata_initialized} metadata files initialized"
                )
            else:
                logger.debug("No metadata population needed - all files already have metadata")

            return result

    @override
    def get_legacy_metadata(self, category: MODEL_REFERENCE_CATEGORY) -> CategoryMetadata:
        """Get legacy format metadata for a specific category.

        Args:
            category: The category to get metadata for.

        Returns:
            CategoryMetadata | None: The legacy metadata, or None if not available.
        """
        return self._metadata_manager.get_legacy_metadata(category)

    @override
    async def get_legacy_metadata_async(self, category: MODEL_REFERENCE_CATEGORY) -> CategoryMetadata:
        """Asynchronously get legacy format metadata for a specific category.

        Args:
            category: The category to get metadata for.

        Returns:
            CategoryMetadata | None: The legacy metadata, or None if not available.
        """
        return self._metadata_manager.get_legacy_metadata(category)

    @override
    def get_metadata(self, category: MODEL_REFERENCE_CATEGORY) -> CategoryMetadata:
        """Get v2 format metadata for a specific category.

        Args:
            category: The category to get metadata for.

        Returns:
            CategoryMetadata | None: The v2 metadata, or None if not available.
        """
        return self._metadata_manager.get_v2_metadata(category)

    @override
    async def get_metadata_async(self, category: MODEL_REFERENCE_CATEGORY) -> CategoryMetadata:
        """Asynchronously get v2 format metadata for a specific category.

        Args:
            category: The category to get metadata for.

        Returns:
            CategoryMetadata | None: The v2 metadata, or None if not available.
        """
        return self._metadata_manager.get_v2_metadata(category)

    @override
    def get_all_legacy_metadata(self) -> dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]:
        """Get legacy format metadata for all categories.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]: Mapping of categories to their legacy metadata.
        """
        return self._metadata_manager.get_all_legacy_metadata()

    @override
    async def get_all_legacy_metadata_async(self) -> dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]:
        """Asynchronously get legacy format metadata for all categories.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]: Mapping of categories to their legacy metadata.
        """
        return self._metadata_manager.get_all_legacy_metadata()

    @override
    def get_all_metadata(self) -> dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]:
        """Get v2 format metadata for all categories.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]: Mapping of categories to their v2 metadata.
        """
        return self._metadata_manager.get_all_v2_metadata()

    @override
    async def get_all_metadata_async(self) -> dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]:
        """Asynchronously get v2 format metadata for all categories.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, CategoryMetadata]: Mapping of categories to their v2 metadata.
        """
        return self._metadata_manager.get_all_v2_metadata()
