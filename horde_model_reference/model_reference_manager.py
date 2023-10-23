import json
from pathlib import Path

import horde_model_reference.path_consts as path_consts
from horde_model_reference.legacy.convert_all_legacy_dbs import convert_all_legacy_model_references
from horde_model_reference.legacy.download_live_legacy_dbs import LegacyReferenceDownloadManager
from horde_model_reference.model_reference_records import (
    MODEL_REFERENCE_TYPE_LOOKUP,
    Generic_ModelReference,
)
from horde_model_reference.path_consts import MODEL_REFERENCE_CATEGORY


class ModelReferenceManager:
    """Class for downloading and reading model reference files."""

    _legacy_reference_download_manager: LegacyReferenceDownloadManager
    _cached_new_references: dict[MODEL_REFERENCE_CATEGORY, Generic_ModelReference | None] = {}

    def __init__(
        self,
        download_and_convert_legacy_dbs: bool = True,
        override_existing: bool = True,
    ) -> None:
        """
        Initialize a new ModelReferenceManager instance.

        Args:
            download_and_convert_legacy_dbs: Whether to download and convert legacy model references.
            override_existing: Whether to override existing model reference files.
        """
        self._legacy_reference_download_manager = LegacyReferenceDownloadManager()
        if download_and_convert_legacy_dbs:
            self.download_and_convert_all_legacy_dbs(override_existing)

    def download_and_convert_all_legacy_dbs(self, override_existing: bool = True) -> bool:
        """
        Download and convert all legacy model reference files.

        Args:
            override_existing: Whether to override existing model reference files.
        """
        self._legacy_reference_download_manager.download_all_legacy_model_references(
            overwrite_existing=override_existing,
        )
        return convert_all_legacy_model_references()

    @property
    def all_legacy_model_reference_file_paths(self) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """
        Get all legacy model reference files.

        Returns:
            A dictionary mapping model reference categories to file paths.
        """
        return self.get_all_legacy_model_reference_file_paths(redownload_all=False)

    def get_all_legacy_model_reference_file_paths(
        self,
        redownload_all: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """
        Get all legacy model reference files.

        Args:
            redownload_all: Whether to redownload all legacy model reference files.

        Returns:
            A dictionary mapping model reference categories to file paths.
        """
        return self._legacy_reference_download_manager.get_all_legacy_model_references(
            redownload_all=redownload_all,
        )

    @property
    def all_model_references(self) -> dict[MODEL_REFERENCE_CATEGORY, Generic_ModelReference | None]:
        """
        Get all model reference files.

        Returns:
            A dictionary mapping model reference categories to file paths. Values of None indicate that the file does
            not exist (failed to download or convert).
        """
        return self.get_all_model_references(redownload_all=False)

    def get_all_model_references(
        self,
        redownload_all: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, Generic_ModelReference | None]:
        """
        Get all model reference files.

        Args:
            redownload_all: Whether to redownload all legacy model reference files.

        Returns:
            A dictionary mapping model reference categories to file paths. Values of None indicate that the file does
            not exist (failed to download or convert).
        """

        if not redownload_all and self._cached_new_references:
            return self._cached_new_references

        if redownload_all:
            self.download_and_convert_all_legacy_dbs()

        all_files: dict[MODEL_REFERENCE_CATEGORY, Path | None] = path_consts.get_all_model_reference_file_paths()

        self._cached_new_references: dict[MODEL_REFERENCE_CATEGORY, Generic_ModelReference | None] = {}

        for category, file_path in all_files.items():
            if file_path is None:
                self._cached_new_references[category] = None
            else:
                with open(file_path) as f:
                    file_contents = f.read()
                file_json: dict = json.loads(file_contents)

                parsed_model = MODEL_REFERENCE_TYPE_LOOKUP[category].model_validate(file_json)

                self._cached_new_references[category] = parsed_model

        return_dict: dict[MODEL_REFERENCE_CATEGORY, Generic_ModelReference | None] = {}
        for reference_type, reference in self._cached_new_references.items():
            if reference is None:
                return_dict[reference_type] = None
                continue
            return_dict[reference_type] = reference.model_copy(deep=True)

        return return_dict
