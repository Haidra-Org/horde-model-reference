"""Constants, especially those to do with paths or network locations, for the horde_model_reference package."""

import os
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORIES

PACKAGE_NAME = "horde_model_reference"
"""The name of this package. Also used as the name of the base folder name for all model reference files."""

BASE_PATH: Path = Path(__file__).parent
"""The base path for all model reference files. Will be based in AIWORKER_CACHE_HOME if set, otherwise will be based in
 this package's install location (IE, in site-packages.)"""

AIWORKER_CACHE_HOME = os.getenv("AIWORKER_CACHE_HOME")
"""The default location for all AI-Horde-Worker cache (model) files."""

if AIWORKER_CACHE_HOME:
    BASE_PATH = Path(AIWORKER_CACHE_HOME).joinpath(PACKAGE_NAME)

LOG_FOLDER: Path = BASE_PATH.joinpath("logs")

LEGACY_REFERENCE_FOLDER_NAME: str = "legacy"
"""The default name of the legacy model reference folder.
If you need the default path, use `LEGACY_REFERENCE_FOLDER`."""

LEGACY_REFERENCE_FOLDER: Path = BASE_PATH.joinpath(LEGACY_REFERENCE_FOLDER_NAME)
"""The default path, starting with BASE_PATH, to the default legacy model reference folder. """

DEFAULT_SHOWCASE_FOLDER_NAME: str = "showcase"
"""The default name of the stable diffusion showcase folder. If you need the path, use `SHOWCASE_FOLDER_PATH`."""

SHOWCASE_FOLDER_PATH: Path = BASE_PATH.joinpath(DEFAULT_SHOWCASE_FOLDER_NAME)
"""The path to the stable diffusion showcase folder."""


def make_all_model_reference_folders():
    """Make all the default model reference folders."""
    if not BASE_PATH.exists():
        BASE_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created base path: {BASE_PATH}")
    if not LOG_FOLDER.exists():
        LOG_FOLDER.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created log folder: {LOG_FOLDER}")
    if not LEGACY_REFERENCE_FOLDER.exists():
        LEGACY_REFERENCE_FOLDER.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created legacy model reference folder: {LEGACY_REFERENCE_FOLDER}")
    if not SHOWCASE_FOLDER_PATH.exists():
        SHOWCASE_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created showcase folder: {SHOWCASE_FOLDER_PATH}")


if os.getenv("HORDE_MODEL_REFERENCE_MAKE_FOLDERS"):
    logger.info("Making all model reference folders.")
    make_all_model_reference_folders()

GITHUB_REPO_OWNER = "Haidra-Org"
GITHUB_REPO_NAME = "AI-Horde-image-model-reference"
GITHUB_REPO_BRANCH = "main"

GITHUB_REPO_URL: str = (
    f"https://raw.githubusercontent.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/{GITHUB_REPO_BRANCH}/"
)
"""The base URL to the live GitHub repo used to power the horde."""

LEGACY_MODEL_GITHUB_URLS = {}
"""A lookup of all the fully qualified file URLs to the given model reference."""

_MODEL_REFERENCE_FILENAMES: dict[MODEL_REFERENCE_CATEGORIES, str] = {}

for category in MODEL_REFERENCE_CATEGORIES:
    filename = f"{category}.json"
    _MODEL_REFERENCE_FILENAMES[category] = filename
    LEGACY_MODEL_GITHUB_URLS[category] = urlparse(GITHUB_REPO_URL + filename).geturl()


def get_model_reference_filename(
    model_reference_category: MODEL_REFERENCE_CATEGORIES,
    *,
    base_path: str | Path | None = None,
) -> str | Path:
    """Return the filename for the given model reference category.

    Args:
        model_reference_category (MODEL_REFERENCE_CATEGORIES): The category of model reference to get the filename for.
        base_path (str | Path | None): If provided, the base path to the model reference file. Defaults to BASE_PATH.

    Returns:
        str: The filename for the given model reference category. If base_path is provided, returns the full path
        from get_model_reference_file_path(...).
    """
    if base_path:
        base_path = Path(base_path)
        if base_path.name == "model_database":
            base_path = BASE_PATH
        return get_model_reference_file_path(model_reference_category, base_path=base_path).resolve()

    return _MODEL_REFERENCE_FILENAMES[model_reference_category]


def get_model_reference_file_path(
    model_reference_category: MODEL_REFERENCE_CATEGORIES,
    *,
    base_path: str | Path = BASE_PATH,
) -> Path:
    """Returns the path to the model reference file for the given model reference category.

    Args:
        model_reference_category (MODEL_REFERENCE_CATEGORIES): The category of model reference to get the filename for.
        basePath (str | Path): If provided, the base path to the model reference file. Defaults to BASE_PATH.

    Returns:
        path:
    """
    if not isinstance(base_path, Path):
        base_path = Path(base_path)
    return base_path.joinpath(_MODEL_REFERENCE_FILENAMES[model_reference_category])
