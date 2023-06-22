import pathlib
from pathlib import Path

import requests
from loguru import logger

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORIES
from horde_model_reference.path_consts import (
    LEGACY_MODEL_GITHUB_URLS,
    LEGACY_REFERENCE_FOLDER,
    LEGACY_REFERENCE_FOLDER_NAME,
    get_model_reference_file_path,
)


class ReferenceDownloadManager:
    proxy_url: str = ""
    """The URL to use as a proxy for downloading files. If empty, no proxy will be used."""

    base_path: str | Path = LEGACY_REFERENCE_FOLDER

    def __init__(
        self,
        *,
        base_path: str | Path = LEGACY_REFERENCE_FOLDER,
        proxy_url: str = "",
    ) -> None:
        self.base_path = base_path
        self.legacy_path = Path(self.base_path).joinpath(LEGACY_REFERENCE_FOLDER_NAME)
        self.proxy_url = proxy_url

    def download_legacy_model_reference(
        self,
        *,
        model_category_name: MODEL_REFERENCE_CATEGORIES,
        override_existing: bool = False,
    ) -> pathlib.Path | None:
        response = requests.get(self.proxy_url + LEGACY_MODEL_GITHUB_URLS[model_category_name])
        target_file_path = get_model_reference_file_path(model_category_name, base_path=self.legacy_path)

        if target_file_path.exists() and not override_existing:
            return None

        target_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file_path, "wb") as f:
            f.write(response.content)
        return target_file_path

    def download_all_legacy_model_references(
        self,
        *,
        override_existing: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORIES, pathlib.Path | None]:
        """Download all legacy model reference files from https://github.com/db0/AI-Horde-image-model-reference.

        Args:
            override_existing (bool, optional): If true, overwrite any existing files. Defaults to False.

        Returns:
            dict[MODEL_REFERENCE_CATEGORIES, Path | None]: The files written, or `None` if that reference failed
        """
        downloaded_files: dict[MODEL_REFERENCE_CATEGORIES, pathlib.Path | None] = {}
        for model_category_name in MODEL_REFERENCE_CATEGORIES:
            downloaded_files[model_category_name] = self.download_legacy_model_reference(
                model_category_name=model_category_name,
                override_existing=override_existing,
            )

        return downloaded_files

    def read_all_legacy_model_references(
        self,
        *,
        redownload_all: bool = False,
    ) -> dict[MODEL_REFERENCE_CATEGORIES, Path | None]:
        """Read all legacy model reference files from disk, optionally redownloading them first."""
        return self.download_all_legacy_model_references(override_existing=redownload_all)


def download_all_models(
    override_existing: bool = False,
    proxy_url: str = "",
) -> dict[MODEL_REFERENCE_CATEGORIES, pathlib.Path | None]:
    reference_dm = ReferenceDownloadManager(proxy_url=proxy_url)
    logger.error("This method is deprecated. Use `download_all_model_references` instead.")
    return reference_dm.download_all_legacy_model_references(override_existing=override_existing)


if __name__ == "__main__":
    reference_download_manager = ReferenceDownloadManager()
    reference_download_manager.download_all_legacy_model_references(override_existing=True)
