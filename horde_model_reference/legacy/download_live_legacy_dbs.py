import pathlib

import requests

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORIES
from horde_model_reference.path_consts import (
    LEGACY_MODEL_REFERENCE_URLS,
    LEGACY_REFERENCE_FOLDER,
    get_model_reference_filename,
)


def download_all_models(override_existing: bool = False) -> dict[MODEL_REFERENCE_CATEGORIES, pathlib.Path]:
    """Download all legacy model reference files from https://github.com/db0/AI-Horde-image-model-reference.

    Args:
        override_existing (bool, optional): If true, overwrite any existing files . Defaults to False.

    Returns:
        dict[MODEL_REFERENCE_CATEGORIES, pathlib.Path]: The downloaded files.
    """
    downloaded_files: dict[MODEL_REFERENCE_CATEGORIES, pathlib.Path] = {}
    for model_category_name, legacy_model_reference_url in LEGACY_MODEL_REFERENCE_URLS.items():
        response = requests.get(legacy_model_reference_url)
        target_file_path = get_model_reference_filename(model_category_name, base_path=LEGACY_REFERENCE_FOLDER)
        if target_file_path.exists() and not override_existing:
            print(f"File already exists: {target_file_path}")
            continue
        target_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file_path, "wb") as f:
            f.write(response.content)
        downloaded_files[model_category_name] = target_file_path
    print(f"Downloaded {len(downloaded_files)} files.")
    return downloaded_files


if __name__ == "__main__":
    download_all_models(True)
