"""Refactored converter classes that use the new Pydantic models from legacy_models.py."""

import glob
import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError
from typing_extensions import override

from horde_model_reference import (
    MODEL_CLASSIFICATION_LOOKUP,
    MODEL_REFERENCE_CATEGORY,
    horde_model_reference_paths,
    path_consts,
)
from horde_model_reference.legacy.classes.legacy_models import (
    LegacyClipRecord,
    LegacyGenericRecord,
    LegacyStableDiffusionRecord,
    LegacyTextGenerationRecord,
)
from horde_model_reference.model_reference_records import (
    CLIPModelRecord,
    DownloadRecord,
    GenericModelRecord,
    GenericModelRecordConfig,
    ImageGenerationModelRecord,
    TextGenerationModelRecord,
)
from horde_model_reference.util import model_name_to_showcase_folder_name

_SLOW_DOWNLOAD_HOST_SUBSTRINGS = ("civitai",)


class BaseLegacyConverter:
    """Base converter for legacy model references using new Pydantic validation."""

    legacy_folder_path: Path
    legacy_database_path: Path
    converted_folder_path: Path
    converted_database_file_path: Path
    model_reference_category: MODEL_REFERENCE_CATEGORY
    model_reference_type: type[LegacyGenericRecord]

    _all_legacy_records: dict[str, LegacyGenericRecord]
    """All validated legacy model records."""
    _all_converted_records: dict[str, GenericModelRecord]
    """All converted model records in the new format."""
    all_validation_errors_log: dict[str, list[str]]
    """All validation errors that occurred during conversion."""
    _host_counter: dict[str, int]
    """Counter for tracking download hosts across all records."""

    debug_mode: bool = False
    log_folder: Path
    dry_run: bool = False
    converted_successfully: bool = False

    def __init__(
        self,
        *,
        legacy_folder_path: str | Path = horde_model_reference_paths.legacy_path,
        target_file_folder: str | Path = horde_model_reference_paths.base_path,
        log_folder: str | Path = horde_model_reference_paths.log_folder,
        model_reference_category: MODEL_REFERENCE_CATEGORY,
        debug_mode: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Initialize the legacy converter.

        Args:
            legacy_folder_path: The legacy database folder.
            target_file_folder: The folder to write the converted database to.
            log_folder: The folder to write the log files to.
            model_reference_category: The category of model reference to convert.
            debug_mode: If true, include extra information in the error log.
            dry_run: If true, don't write out the converted database or any logs.
        """
        self._initialize()

        self.model_reference_category = model_reference_category
        self.model_reference_type = LegacyGenericRecord
        if model_reference_category == MODEL_REFERENCE_CATEGORY.image_generation:
            self.model_reference_type = LegacyStableDiffusionRecord
        elif model_reference_category == MODEL_REFERENCE_CATEGORY.clip:
            self.model_reference_type = LegacyClipRecord
        elif model_reference_category == MODEL_REFERENCE_CATEGORY.text_generation:
            self.model_reference_type = LegacyTextGenerationRecord

        self.legacy_folder_path = Path(legacy_folder_path)
        self.legacy_database_path = horde_model_reference_paths.get_model_reference_file_path(
            model_reference_category=model_reference_category,
            base_path=legacy_folder_path,
        )
        self.converted_folder_path = Path(target_file_folder)
        self.converted_database_file_path = horde_model_reference_paths.get_model_reference_file_path(
            model_reference_category=model_reference_category,
            base_path=target_file_folder,
        )
        self.debug_mode = debug_mode
        self.log_folder = Path(log_folder)
        self.dry_run = dry_run

    def _initialize(self) -> None:
        """Initialize the converter, allowing re-conversion if applicable."""
        self._all_legacy_records = {}
        self._all_converted_records = {}
        self.all_validation_errors_log = {}
        self._host_counter = {}
        self.converted_successfully = False

    def convert_to_new_format(self) -> dict[str, GenericModelRecord]:
        """Convert the legacy model reference to the new format.

        Returns:
            The converted model records in the new format.
        """
        if self.converted_successfully:
            self._initialize()

        self.pre_parse_records()
        self._load_and_validate_legacy_records()
        self._convert_legacy_to_new_format()
        self.post_parse_records()
        self.write_out_validation_errors()
        self.write_out_records()

        self.converted_successfully = True

        return self._all_converted_records

    def _load_and_validate_legacy_records(self) -> None:
        """Load and validate all legacy records using Pydantic models."""
        with open(self.legacy_database_path) as legacy_model_reference_file:
            raw_legacy_json_data: dict[str, dict[str, Any]] = json.load(legacy_model_reference_file)

        for model_record_key, model_record_contents in raw_legacy_json_data.items():
            issues: list[str] = []
            validation_context = {
                "issues": issues,
                "model_key": model_record_key,
                "debug_mode": self.debug_mode,
                "category": self.model_reference_category,
                "host_counter": self._host_counter,
            }

            # Add existing showcase files to context for stable diffusion
            if hasattr(self, "existing_showcase_files"):
                validation_context["existing_showcase_files"] = self.existing_showcase_files

            try:
                legacy_record = self.model_reference_type.model_validate(
                    model_record_contents,
                    context=validation_context,
                )
                self._all_legacy_records[model_record_key] = legacy_record

                if issues:
                    for issue in issues:
                        self.add_validation_error_to_log(model_record_key=model_record_key, error=issue)
            except ValidationError as e:
                error = f"CRITICAL: Error parsing {model_record_key}:\n{e}"
                self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                raise

    def _convert_legacy_to_new_format(self) -> None:
        """Convert validated legacy records to the new format."""
        for model_key, legacy_record in self._all_legacy_records.items():
            try:
                converted_record = self._convert_single_record(legacy_record)
                self._all_converted_records[model_key] = converted_record
            except Exception as e:
                error = f"Failed to convert {model_key}: {e}"
                self.add_validation_error_to_log(model_record_key=model_key, error=error)
                raise

    """legacy config example
        "config": {
            "files": [
                {
                    "path": "model_2_1.ckpt",
                    "md5sum": "ce8ee5c53acb3a6540b44a71276c3d01",
                    "sha256sum": "ad2a33c361c1f593c4a1fb32ea81afce2b5bb7d1983c6b94793a26a3b54b08a0"
                },
                {
                    "path": "v2-inference-v.yaml"
                }
            ],
            "download": [
                {
                    "file_name": "model_2_1.ckpt",
                    "file_path": "",
                    "file_url": "https://huggingface.co/mirroring/horde_models/resolve/main/model_2_1.ckpt?download=true"
                }
            ]
        },
    """

    def _convert_model_record_config(self, legacy_record: LegacyGenericRecord) -> GenericModelRecordConfig:
        """Convert the config section of a legacy record to the new format."""
        download_records: dict[str, DownloadRecord] = {}

        for file_entry in legacy_record.config.files:
            if file_entry.path and "yaml" not in file_entry.path.lower():
                download_records[file_entry.path] = DownloadRecord(
                    file_name=file_entry.path,
                    file_url="",
                    sha256sum=file_entry.sha256sum if file_entry.sha256sum else "FIXME",
                    file_type=None,
                    known_slow_download=any(
                        slow_url in file_entry.path.lower() for slow_url in _SLOW_DOWNLOAD_HOST_SUBSTRINGS
                    ),
                )

        for download_entry in legacy_record.config.download:
            if download_entry.file_name in download_records:

                download_records[download_entry.file_name].file_url = download_entry.file_url or ""
            else:
                raise ValueError(f"Unknown download entry: {download_entry.file_name}")

        return GenericModelRecordConfig(download=list(download_records.values()))

    def _convert_single_record(
        self,
        legacy_record: LegacyGenericRecord,
    ) -> GenericModelRecord:
        """Convert a single legacy record to the new format.

        Override this in subclasses for category-specific conversion.
        """
        model_record_config = self._convert_model_record_config(legacy_record)

        return GenericModelRecord(
            name=legacy_record.name,
            description=legacy_record.description,
            version=legacy_record.version,
            config=model_record_config,
            model_classification=MODEL_CLASSIFICATION_LOOKUP[self.model_reference_category],
        )

    def pre_parse_records(self) -> None:
        """Override to perform category-specific pre-parsing."""

    def post_parse_records(self) -> None:
        """Override to perform category-specific post-parsing."""

    def write_out_records(self) -> None:
        """Write out the converted records."""
        if self.dry_run:
            return
        # Serialize the converted records to a canonical JSON string first so we can
        # compare with the on-disk file and avoid rewriting (which changes mtime).
        final_serialized = json.dumps(
            self._all_converted_records,
            indent=4,
            default=lambda o: o.model_dump(
                exclude_none=True,
                exclude_unset=False,
                by_alias=True,
            ),
        )
        # keep trailing newline for consistency with other writers
        final_serialized = final_serialized + "\n"

        target_path = Path(self.converted_database_file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # If the file already exists and the content is identical, skip writing to
        # preserve the existing mtime and avoid invalidating caches that rely on it.
        try:
            if target_path.exists():
                existing = target_path.read_text()
                if existing == final_serialized:
                    logger.debug(f"No change to converted file {target_path}, skipping write.")
                    return
        except Exception:
            # If we can't read the existing file for any reason, continue and overwrite.
            pass

        # Write atomically: write to a temporary file in the same directory and replace.
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        tmp_path.write_text(final_serialized)
        tmp_path.replace(target_path)

    def get_records(self) -> dict[str, GenericModelRecord]:
        """Return the converted records."""
        return self._all_converted_records

    def add_validation_error_to_log(
        self,
        *,
        model_record_key: str,
        error: str,
    ) -> None:
        """Add a validation error to the log."""
        if model_record_key not in self.all_validation_errors_log:
            self.all_validation_errors_log[model_record_key] = []
        self.all_validation_errors_log[model_record_key].append(error)

        if self.debug_mode:
            logger.debug(f"{model_record_key} has error: {error}")

    def write_out_validation_errors(self) -> None:
        """Write out the validation errors."""
        if self.dry_run or not self.debug_mode:
            return

        log_file = self.log_folder.joinpath(self.model_reference_category + ".log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as validation_errors_log_file:
            validation_errors_log_file.write(
                json.dumps(
                    self.all_validation_errors_log,
                    indent=4,
                ),
            )


class LegacyStableDiffusionConverter(BaseLegacyConverter):
    """Converter for legacy Stable Diffusion model reference records."""

    showcase_glob_pattern: str = "horde_model_reference/showcase/*"
    all_baseline_categories: dict[str, int]
    all_styles: dict[str, int]
    all_tags: dict[str, int]
    existing_showcase_files: dict[str, list[str]]

    def __init__(
        self,
        *,
        legacy_folder_path: str | Path = horde_model_reference_paths.legacy_path,
        target_file_folder: str | Path = horde_model_reference_paths.base_path,
        debug_mode: bool = False,
    ) -> None:
        """Initialize the Stable Diffusion converter."""
        super().__init__(
            legacy_folder_path=legacy_folder_path,
            target_file_folder=target_file_folder,
            model_reference_category=MODEL_REFERENCE_CATEGORY.image_generation,
            debug_mode=debug_mode,
        )
        self._sd_initialize()

    def _sd_initialize(self) -> None:
        """Initialize SD-specific tracking dictionaries."""
        self.all_baseline_categories = {}
        self.all_styles = {}
        self.all_tags = {}
        self.existing_showcase_files = {}

    @override
    def pre_parse_records(self) -> None:
        existing_showcase_folders = glob.glob(self.showcase_glob_pattern, recursive=True)
        self.existing_showcase_files = self.get_existing_showcases(existing_showcase_folders)

    @override
    def _convert_single_record(
        self,
        legacy_record: LegacyGenericRecord,
    ) -> ImageGenerationModelRecord:
        """Convert a single legacy Stable Diffusion record to the new format."""
        if not isinstance(legacy_record, LegacyStableDiffusionRecord):
            raise TypeError(f"Expected {legacy_record.name} to be a LegacyStableDiffusionRecord.")

        if legacy_record.baseline:
            self.all_baseline_categories[legacy_record.baseline] = (
                self.all_baseline_categories.get(legacy_record.baseline, 0) + 1
            )
        if legacy_record.style:
            self.all_styles[legacy_record.style] = self.all_styles.get(legacy_record.style, 0) + 1
        if legacy_record.tags:
            for tag in legacy_record.tags:
                self.all_tags[tag] = self.all_tags.get(tag, 0) + 1

        model_record_config = self._convert_model_record_config(legacy_record)

        return ImageGenerationModelRecord(
            name=legacy_record.name,
            description=legacy_record.description,
            version=legacy_record.version,
            config=model_record_config,
            inpainting=legacy_record.inpainting,
            baseline=legacy_record.baseline,
            optimization=legacy_record.optimization,
            tags=legacy_record.tags or [],
            showcases=legacy_record.showcases or [],
            min_bridge_version=legacy_record.min_bridge_version,
            trigger=legacy_record.trigger or [],
            homepage=legacy_record.homepage,
            nsfw=legacy_record.nsfw,
            style=legacy_record.style,
            requirements=legacy_record.requirements,
            size_on_disk_bytes=legacy_record.size_on_disk_bytes,
            model_classification=MODEL_CLASSIFICATION_LOOKUP[self.model_reference_category],
        )

    @override
    def post_parse_records(self) -> None:
        super().post_parse_records()

        # Create showcase folder
        # for model_key in self._all_converted_records:
        #     expected_showcase_foldername = model_name_to_showcase_folder_name(model_key)
        #     self.create_showcase_folder(expected_showcase_foldername)

        final_on_disk_showcase_folders = glob.glob(self.showcase_glob_pattern, recursive=True)
        for folder in final_on_disk_showcase_folders:
            parsed_folder = Path(folder)

            if parsed_folder.is_file():
                continue

            if not any(parsed_folder.iterdir()):
                error = f"showcase folder '{parsed_folder.name}' is empty."
                self.add_validation_error_to_log(model_record_key=parsed_folder.name, error=error)

        final_on_disk_showcase_folders_names = [
            Path(folder).name for folder in final_on_disk_showcase_folders if Path(folder).is_dir()
        ]
        final_expected_showcase_folders = [
            model_name_to_showcase_folder_name(model_name) for model_name in self._all_converted_records
        ]

        for folder in final_on_disk_showcase_folders_names:
            if folder not in final_expected_showcase_folders:
                error = f"folder '{folder}' is not in the model records."
                self.add_validation_error_to_log(model_record_key=folder, error=error)

        if self.debug_mode:
            logger.debug(f"{self.all_styles=}")
            logger.debug(f"{self.all_baseline_categories=}")
            logger.debug(f"{self.all_tags=}")
            logger.debug(f"{self._host_counter=}")
            logger.info(f"Total number of models: {len(self._all_converted_records)}")
            logger.info(f"Total number of showcase folders: {len(final_on_disk_showcase_folders_names)}")
            logger.info(f"Total number of models with validation issues: {len(self.all_validation_errors_log)}")

    @override
    def write_out_records(self) -> None:
        sanity_check: dict[str, ImageGenerationModelRecord] = {
            key: value
            for key, value in self._all_converted_records.items()
            if isinstance(value, ImageGenerationModelRecord)
        }
        if len(sanity_check) != len(self._all_converted_records):
            raise ValueError("CRITICAL: Not all records are of the correct type.")

        if self.dry_run:
            return

        final_converted_model_reference = json.dumps(
            self._all_converted_records,
            indent=4,
            default=lambda o: o.model_dump(
                exclude_none=True,
                exclude_unset=True,
                exclude_defaults=True,
                by_alias=True,
            ),
        )
        final_converted_model_reference = final_converted_model_reference + "\n"

        target_path = Path(self.converted_database_file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if target_path.exists():
                existing = target_path.read_text()
                if existing == final_converted_model_reference:
                    logger.debug(f"No change to converted file {target_path}, skipping write.")
                    return
        except Exception:
            pass

        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        tmp_path.write_text(final_converted_model_reference)
        tmp_path.replace(target_path)

        logger.debug(f"Converted database written to: {self.converted_database_file_path}")

    def get_existing_showcases(
        self,
        existing_showcase_folders: list[str],
    ) -> dict[str, list[str]]:
        """Return a dictionary of existing showcase files, keyed by showcase folder name."""
        existing_showcase_files: dict[str, list[str]] = {}
        for showcase_folder in existing_showcase_folders:
            model_showcase_files = glob.glob(str(Path(showcase_folder).joinpath("*")), recursive=True)
            model_showcase_folder_name = model_name_to_showcase_folder_name(Path(showcase_folder).name)

            existing_showcase_files[model_showcase_folder_name] = model_showcase_files

        return existing_showcase_files

    def create_showcase_folder(self, showcase_foldername: str) -> None:
        """Create a showcase folder with the given name."""
        if showcase_foldername not in self.existing_showcase_files:
            self.existing_showcase_files[showcase_foldername] = []

        newFolder = self.converted_folder_path.joinpath(path_consts.DEFAULT_SHOWCASE_FOLDER_NAME)
        newFolder = newFolder.joinpath(showcase_foldername)
        newFolder.mkdir(parents=True, exist_ok=True)


class LegacyClipConverter(BaseLegacyConverter):
    """Converter for legacy CLIP model reference records."""

    def __init__(
        self,
        *,
        legacy_folder_path: str | Path = horde_model_reference_paths.legacy_path,
        target_file_folder: str | Path = horde_model_reference_paths.base_path,
        debug_mode: bool = False,
    ) -> None:
        """Initialize the legacy CLIP converter."""
        super().__init__(
            legacy_folder_path=legacy_folder_path,
            target_file_folder=target_file_folder,
            model_reference_category=MODEL_REFERENCE_CATEGORY.clip,
            debug_mode=debug_mode,
        )

    @override
    def _convert_single_record(
        self,
        legacy_record: LegacyGenericRecord,
    ) -> CLIPModelRecord:
        """Convert a single legacy CLIP record to the new format."""
        if not isinstance(legacy_record, LegacyClipRecord):
            raise TypeError(f"Expected {legacy_record.name} to be a LegacyClipRecord.")

        model_record_config = self._convert_model_record_config(legacy_record)

        return CLIPModelRecord(
            name=legacy_record.name,
            description=legacy_record.description,
            version=legacy_record.version,
            config=model_record_config,
            pretrained_name=legacy_record.pretrained_name,
            model_classification=MODEL_CLASSIFICATION_LOOKUP[self.model_reference_category],
        )


class LegacyTextGenerationConverter(BaseLegacyConverter):
    """Converter for legacy text generation model reference records."""

    def __init__(
        self,
        *,
        legacy_folder_path: str | Path = horde_model_reference_paths.legacy_path,
        target_file_folder: str | Path = horde_model_reference_paths.base_path,
        debug_mode: bool = False,
    ) -> None:
        """Initialize the legacy text generation converter."""
        super().__init__(
            legacy_folder_path=legacy_folder_path,
            target_file_folder=target_file_folder,
            model_reference_category=MODEL_REFERENCE_CATEGORY.text_generation,
            debug_mode=debug_mode,
        )

    @override
    def _convert_single_record(
        self,
        legacy_record: LegacyGenericRecord,
    ) -> TextGenerationModelRecord:
        """Convert a single legacy text generation record to the new format."""
        if not isinstance(legacy_record, LegacyTextGenerationRecord):
            raise TypeError(f"Expected {legacy_record.name} to be a LegacyTextGenerationRecord.")

        model_record_config = self._convert_model_record_config(legacy_record)

        return TextGenerationModelRecord(
            name=legacy_record.name,
            description=legacy_record.description,
            version=legacy_record.version,
            config=model_record_config,
            baseline=legacy_record.baseline,
            parameters=legacy_record.parameters or 0,
            nsfw=legacy_record.nsfw or False,
            style=legacy_record.style,
            display_name=legacy_record.display_name,
            url=legacy_record.url,
            tags=legacy_record.tags or [],
            settings=legacy_record.settings,
            model_classification=MODEL_CLASSIFICATION_LOOKUP[self.model_reference_category],
        )
