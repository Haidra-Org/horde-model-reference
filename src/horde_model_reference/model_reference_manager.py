from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar

from loguru import logger

import horde_model_reference.path_consts as path_consts
from horde_model_reference.legacy import LegacyReferenceDownloadManager
from horde_model_reference.meta_consts import MODEL_CLASSIFICATION_LOOKUP, MODEL_REFERENCE_CATEGORY
from horde_model_reference.model_reference_records import (
    KNOWN_MODEL_REFERENCE_INSTANCES,
    MODEL_REFERENCE_CATEGORY_TYPE_LOOKUP,
    Generic_ModelRecord,
)

MR = TypeVar("MR", bound=Generic_ModelRecord)


class ModelReferenceManager:
    """Class for downloading and reading model reference files."""

    legacy_reference_download_manager: LegacyReferenceDownloadManager
    _cached_file_json: dict[MODEL_REFERENCE_CATEGORY, dict[Any, Any] | None]

    _instance: ModelReferenceManager | None = None
    _lazy_mode: bool = True

    _lock: RLock = RLock()

    def __new__(
        cls,
        *,
        override_existing: bool = False,
        lazy_mode: bool = True,
        base_path: str | Path = path_consts.BASE_PATH,
        proxy_url: str = path_consts.HORDE_PROXY_URL_BASE,
    ) -> ModelReferenceManager:
        """Create a new instance of ModelReferenceManager.

        Use the singleton pattern to ensure only one instance exists to avoid multiple downloads and conversions.
        """
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)

                cls._instance._cached_file_json = {}
                cls._instance.legacy_reference_download_manager = LegacyReferenceDownloadManager(
                    base_path=base_path,
                    proxy_url=proxy_url,
                )
                cls._instance._lazy_mode = lazy_mode
                if not lazy_mode:
                    cls._instance.download_and_convert_legacy_references(override_existing=override_existing)

                if lazy_mode and override_existing:
                    logger.warning(
                        "lazy_mode and override_existing are both enabled. "
                        "lazy_mode prevents downloading model references when initializing "
                        "even if override_existing is True.",
                    )

        return cls._instance

    def download_and_convert_legacy_references(
        self,
        override_existing: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Download and convert all legacy model reference files.

        Args:
            override_existing (bool, optional): Whether to override existing model reference files. Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, Path | None]: A mapping of model reference categories to file paths.
        """
        return self.legacy_reference_download_manager.download_all_legacy_model_references(
            overwrite_existing=override_existing,
        )

    @property
    def all_model_references(self) -> dict[MODEL_REFERENCE_CATEGORY, KNOWN_MODEL_REFERENCE_INSTANCES | None]:
        """Get all model references.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, KNOWN_MODEL_REFERENCE_INSTANCES | None]: A mapping of model reference
            categories to their corresponding model reference objects.
        """
        return self.get_all_model_references(override_existing=False)

    def file_json_to_model_reference(
        self,
        category: MODEL_REFERENCE_CATEGORY,
        file_json: dict[str, Any] | None,
    ) -> KNOWN_MODEL_REFERENCE_INSTANCES | None:
        """Convert a file JSON object to a model reference.

        Args:
            category (MODEL_REFERENCE_CATEGORY): The category of the model reference.
            file_json (dict): The JSON object representing the model reference.

        Returns:
            KNOWN_MODEL_REFERENCE_INSTANCES | None: The model reference object, or None if conversion failed.
        """
        if file_json is None:
            logger.warning(f"File JSON is None for {category}.")
            return None

        try:
            for model_value in file_json.values():
                if "model_classification" not in model_value:
                    model_value["model_classification"] = MODEL_CLASSIFICATION_LOOKUP[category]

            return MODEL_REFERENCE_CATEGORY_TYPE_LOOKUP[category].model_validate(file_json)
        except Exception as e:
            logger.exception(f"Failed to convert file JSON to model reference for {category}: {e}")
            return None

    def _get_all_cached_model_references(
        self,
    ) -> dict[MODEL_REFERENCE_CATEGORY, KNOWN_MODEL_REFERENCE_INSTANCES | None]:
        """Get all cached model references.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, KNOWN_MODEL_REFERENCE_INSTANCES | None]: A mapping of model reference
                categories to their corresponding model reference objects.
        """
        return_dict = {}
        with self._lock:
            for category, file_json in self._cached_file_json.items():
                model_reference = self.file_json_to_model_reference(category, file_json)
                return_dict[category] = model_reference

        logger.debug(f"Returning {len(return_dict)} cached model references.")
        return return_dict

    def get_all_model_references(
        self,
        override_existing: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, KNOWN_MODEL_REFERENCE_INSTANCES | None]:
        """Get a mapping of model reference categories labels and the corresponding model reference objects.

        Args:
            override_existing (bool, optional): Whether to force a redownload of all model reference files.
                Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, KNOWN_MODEL_REFERENCE_INSTANCES | None]: A mapping of model reference
                categories to their corresponding model reference objects.
        """
        with self._lock:
            if not override_existing and self._cached_file_json:
                logger.debug("Using cached model references.")
                return self._get_all_cached_model_references()

            self.download_and_convert_legacy_references(override_existing=override_existing)

            all_files: dict[MODEL_REFERENCE_CATEGORY, Path | None] = path_consts.get_all_model_reference_file_paths()

            for category, file_path in all_files.items():
                if file_path is None:
                    self._cached_file_json[category] = None
                    continue

                if not file_path.exists():
                    logger.warning(
                        f"Model reference file for {category} does not exist at {file_path}.",
                    )
                    self._cached_file_json[category] = None
                    continue

                with open(file_path) as f:
                    file_contents = f.read()
                try:
                    file_json = json.loads(file_contents)
                    self._cached_file_json[category] = file_json
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON for {category} from {file_path}: {e}")
                    self._cached_file_json[category] = None

            return self._get_all_cached_model_references()
