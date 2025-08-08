from __future__ import annotations

import json
import urllib.parse
from pathlib import Path
from threading import RLock

import requests
from loguru import logger

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.path_consts import (
    BASE_PATH,
    HORDE_PROXY_URL_BASE,
    LEGACY_MODEL_GITHUB_URLS,
    LEGACY_REFERENCE_FOLDER_NAME,
    get_model_reference_file_path,
)


class LegacyReferenceDownloadManager:
    """Class for downloading and reading the AI-Horde legacy model reference files.

    See the `path_consts.py` file for the exact URLs and paths.
    """

    base_path: str | Path = BASE_PATH
    """The base path to use for all file operations."""
    legacy_path: Path
    """The path to the legacy reference folder."""

    proxy_url: str = HORDE_PROXY_URL_BASE
    """The URL to use as a proxy for downloading files. If empty, no proxy will be used."""

    _cached_file_locations: dict[MODEL_REFERENCE_CATEGORY, Path | None]
    _times_downloaded: dict[MODEL_REFERENCE_CATEGORY, int]

    _instance: LegacyReferenceDownloadManager | None = None
    _lock: RLock = RLock()

    def __new__(
        cls,
        *,
        base_path: str | Path = BASE_PATH,
        proxy_url: str = HORDE_PROXY_URL_BASE,
    ) -> LegacyReferenceDownloadManager:
        """Return the existing instance of LegacyReferenceDownloadManager, or create a new one if it doesn't exist."""
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance.base_path = base_path
                cls._instance.legacy_path = Path(cls._instance.base_path).joinpath(LEGACY_REFERENCE_FOLDER_NAME)
                cls._instance.proxy_url = proxy_url
                cls._instance._cached_file_locations = {}
                cls._instance._times_downloaded = {}

                for model_reference_category in MODEL_REFERENCE_CATEGORY:
                    file_path = get_model_reference_file_path(
                        model_reference_category,
                        base_path=cls._instance.legacy_path,
                    )

                    if file_path.exists():
                        cls._instance._cached_file_locations[model_reference_category] = file_path
                    else:
                        cls._instance._cached_file_locations[model_reference_category] = None

                    cls._instance._times_downloaded[model_reference_category] = 0

        return cls._instance

    def download_legacy_model_reference(
        self,
        *,
        model_category_name: MODEL_REFERENCE_CATEGORY,
        override_existing: bool = False,
    ) -> Path | None:
        """Download a legacy model reference file from `LEGACY_MODEL_GITHUB_URLS`.

        Args:
            model_category_name (MODEL_REFERENCE_CATEGORY): The model category to download.
            override_existing (bool, optional): If true, overwrite any existing files. Defaults to False.

        Returns:
            Path | None: The path to the downloaded file, or None if the download failed for any reason.
        """
        target_file_path = get_model_reference_file_path(model_category_name, base_path=self.legacy_path)

        with self._lock:
            if target_file_path.exists() and not override_existing:
                logger.debug(f"File {target_file_path} already exists, skipping download.")
                return target_file_path

            target_url = urllib.parse.urljoin(self.proxy_url, LEGACY_MODEL_GITHUB_URLS[model_category_name])
            response = requests.get(target_url)

            if response.status_code != 200:
                logger.error(f"Failed to download {model_category_name} reference file.")
                return None

            try:
                json.loads(response.content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse {model_category_name} reference file as JSON.")
                return None

            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_file_path, "wb") as f:
                f.write(response.content)

            self._times_downloaded[model_category_name] += 1

            if self._times_downloaded[model_category_name] > 1:
                logger.debug(
                    f"Downloaded {model_category_name} reference file {self._times_downloaded[model_category_name]} "
                    "times.",
                )

            logger.info(f"Downloaded {model_category_name} reference file to {target_file_path}.")
            self._cached_file_locations[model_category_name] = target_file_path

        return target_file_path

    def download_all_legacy_model_references(
        self,
        *,
        overwrite_existing: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Download all legacy model reference files from `LEGACY_MODEL_GITHUB_URLS`.

        See the module `horde_model_reference.path_consts` for details.

        Args:
            overwrite_existing (bool, optional): If true, overwrite any existing files. Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, Path | None]: The files written, or `None` if that reference failed
        """
        downloaded_files: dict[MODEL_REFERENCE_CATEGORY, Path | None] = {}
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
        """Read all legacy model reference files from disk, optionally forcing a redownload first.

        Args:
            redownload_all (bool, optional): If true, redownload all files even if they exist locally.
                Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORY, Path | None]: The paths to the legacy model reference files, or None if a
                file could not be read.
        """
        with self._lock:
            self._cached_file_locations = self.download_all_legacy_model_references(overwrite_existing=redownload_all)

            return self._cached_file_locations.copy()

    def is_downloaded(
        self,
        model_category_name: MODEL_REFERENCE_CATEGORY,
    ) -> bool:
        """Check if the given model reference category has been downloaded."""
        with self._lock:
            model_path = self._cached_file_locations.get(model_category_name)
            if model_path is None:
                return False

            if model_path.exists():
                return True

            logger.warning(
                f"Model reference file for {model_category_name} does not exist at {model_path}, but was previously "
                "downloaded. Invalidating cache for this model reference.",
            )
            self._cached_file_locations[model_category_name] = None
            return False

    def is_all_downloaded(self) -> bool:
        """Check if all model reference categories have been downloaded."""
        with self._lock:
            if not self._cached_file_locations:
                return False

            return all(self.is_downloaded(category) for category in MODEL_REFERENCE_CATEGORY)
