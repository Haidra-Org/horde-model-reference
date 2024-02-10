import json
import pathlib
from pathlib import Path

import requests
from loguru import logger

from horde_model_reference.legacy.convert_all_legacy_dbs import convert_all_legacy_model_references
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.path_consts import (
    BASE_PATH,
    HORDE_PROXY_URL_BASE,
    LEGACY_MODEL_GITHUB_URLS,
    LEGACY_REFERENCE_FOLDER_NAME,
    get_model_reference_file_path,
)


class LegacyReferenceDownloadManager:
    base_path: str | Path = BASE_PATH
    """The base path to use for all file operations."""
    legacy_path: Path
    """The path to the legacy reference folder."""

    proxy_url: str = HORDE_PROXY_URL_BASE
    """The URL to use as a proxy for downloading files. If empty, no proxy will be used."""

    _cached_file_locations: dict[MODEL_REFERENCE_CATEGORY, pathlib.Path | None] | None = None

    def __init__(
        self,
        *,
        base_path: str | Path = BASE_PATH,
        proxy_url: str = HORDE_PROXY_URL_BASE,
    ) -> None:
        self.base_path = base_path
        self.legacy_path = Path(self.base_path).joinpath(LEGACY_REFERENCE_FOLDER_NAME)
        self.proxy_url = proxy_url

    def download_legacy_model_reference(
        self,
        *,
        model_category_name: MODEL_REFERENCE_CATEGORY,
        override_existing: bool = False,
    ) -> pathlib.Path | None:
        response = requests.get(self.proxy_url + LEGACY_MODEL_GITHUB_URLS[model_category_name])
        if response.status_code != 200:
            logger.error(f"Failed to download {model_category_name} reference file.")
            return None

        try:
            json.loads(response.content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse {model_category_name} reference file as JSON.")
            return None

        target_file_path = get_model_reference_file_path(model_category_name, base_path=self.legacy_path)

        if target_file_path.exists() and not override_existing:
            logger.debug(f"File {target_file_path} already exists, skipping download.")
            return None

        target_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file_path, "wb") as f:
            f.write(response.content)
        return target_file_path

    def download_all_legacy_model_references(
        self,
        *,
        overwrite_existing: bool = True,
    ) -> dict[MODEL_REFERENCE_CATEGORY, pathlib.Path | None]:
        """Download all legacy model reference files from https://github.com/db0/AI-Horde-image-model-reference.

        Args:
            override_existing (bool, optional): If true, overwrite any existing files. Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, Path | None]: The files written, or `None` if that reference failed
        """
        downloaded_files: dict[MODEL_REFERENCE_CATEGORY, pathlib.Path | None] = {}
        for model_category_name in MODEL_REFERENCE_CATEGORY:
            downloaded_files[model_category_name] = self.download_legacy_model_reference(
                model_category_name=model_category_name,
                override_existing=overwrite_existing,
            )

        return downloaded_files

    def get_all_legacy_model_references(
        self,
        *,
        redownload_all: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Read all legacy model reference files from disk, optionally redownloading them first."""
        if not redownload_all and self._cached_file_locations:
            return self._cached_file_locations

        self._cached_file_locations = self.download_all_legacy_model_references(overwrite_existing=redownload_all)

        return self._cached_file_locations

    def convert_legacy_references(self):
        """Convert all legacy model reference files to the new format."""
        convert_all_legacy_model_references(
            base_path=self.base_path,
            legacy_path=self.legacy_path,
        )
