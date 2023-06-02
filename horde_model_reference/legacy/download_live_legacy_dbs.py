import pathlib

import requests

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORIES
from horde_model_reference.path_consts import (
    LEGACY_MODEL_GITHUB_URLS,
    LEGACY_REFERENCE_FOLDER,
    get_model_reference_file_path,
)


def download_model_reference(
    model_category_name: MODEL_REFERENCE_CATEGORIES,
    override_existing: bool = False,
    *,
    proxy_url: str = "",
) -> pathlib.Path:
    response = requests.get(proxy_url + LEGACY_MODEL_GITHUB_URLS[model_category_name])
    target_file_path = get_model_reference_file_path(model_category_name, base_path=LEGACY_REFERENCE_FOLDER)

    if target_file_path.exists() and not override_existing:
        return None

    target_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_file_path, "wb") as f:
        f.write(response.content)
    downloaded_files[model_category_name] = target_file_path


def download_all_models(
    override_existing: bool = False,
    *,
    proxy_url: str = "",
) -> dict[MODEL_REFERENCE_CATEGORIES, pathlib.Path]:
    """Download all legacy model reference files from https://github.com/db0/AI-Horde-image-model-reference.

    Args:
        override_existing (bool, optional): If true, overwrite any existing files . Defaults to False.

    Returns:
        dict[MODEL_REFERENCE_CATEGORIES, pathlib.Path]: The downloaded files.
    """
    downloaded_files: dict[MODEL_REFERENCE_CATEGORIES, pathlib.Path] = {}
    for model_category_name in MODEL_REFERENCE_CATEGORIES:
        downloaded_files.update(
            (
                model_category_name,
                download_model_reference(
                    model_category_name,
                    override_existing=override_existing,
                    proxy_url=proxy_url,
                ),
            ),
        )

    print(f"Downloaded {len(downloaded_files)} files.")
    return downloaded_files


if __name__ == "__main__":
    download_all_models(True)
