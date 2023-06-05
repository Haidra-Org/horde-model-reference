"""Constants, especially those to do with paths or network locations, for the horde_model_reference package."""

from pathlib import Path
from urllib.parse import urlparse

from horde_model_reference import MODEL_REFERENCE_CATEGORIES

PACKAGE_NAME = "horde_model_reference"

GITHUB_REPO_OWNER = "Haidra-Org"
GITHUB_REPO_NAME = "AI-Horde-image-model-reference"
GITHUB_REPO_BRANCH = "main"

GITHUB_REPO_URL: str = (
    f"https://raw.githubusercontent.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/{GITHUB_REPO_BRANCH}/"
)
"""The base URL to the live GitHub repo used to power the horde."""

LEGACY_MODEL_GITHUB_URLS = {
    MODEL_REFERENCE_CATEGORIES.BLIP: urlparse(GITHUB_REPO_URL + "blip.json").geturl(),
    MODEL_REFERENCE_CATEGORIES.CLIP: urlparse(GITHUB_REPO_URL + "clip.json").geturl(),
    MODEL_REFERENCE_CATEGORIES.CODEFORMER: urlparse(GITHUB_REPO_URL + "codeformer.json").geturl(),
    MODEL_REFERENCE_CATEGORIES.CONTROLNET: urlparse(GITHUB_REPO_URL + "controlnet.json").geturl(),
    MODEL_REFERENCE_CATEGORIES.ESRGAN: urlparse(GITHUB_REPO_URL + "esrgan.json").geturl(),
    MODEL_REFERENCE_CATEGORIES.GFPGAN: urlparse(GITHUB_REPO_URL + "gfpgan.json").geturl(),
    MODEL_REFERENCE_CATEGORIES.SAFETY_CHECKER: urlparse(GITHUB_REPO_URL + "safety_checker.json").geturl(),
    MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION: urlparse(GITHUB_REPO_URL + "stable_diffusion.json").geturl(),
}
"""A lookup of all the fully qualified URLs to the live model reference files."""


BASE_PATH: Path = Path(__file__).parent
"""The base path to the image database folder. Should end in `horde_model_reference/`."""

LEGACY_REFERENCE_FOLDER: Path = BASE_PATH.joinpath("legacy")
"""The path to the legacy model reference folder."""


DEFAULT_SHOWCASE_FOLDER_NAME: str = "showcase"
"""The default name of the stable diffusion showcase folder. If you need the path, use `SHOWCASE_FOLDER_PATH`."""


SHOWCASE_FOLDER_PATH: Path = BASE_PATH.joinpath(DEFAULT_SHOWCASE_FOLDER_NAME)
"""The path to the stable diffusion showcase folder."""


_MODEL_REFERENCE_FILENAMES: dict[MODEL_REFERENCE_CATEGORIES, str] = {}

for category in MODEL_REFERENCE_CATEGORIES:
    _MODEL_REFERENCE_FILENAMES[category] = f"{category}.json"


def get_model_reference_filename(
    model_reference_category: MODEL_REFERENCE_CATEGORIES,
) -> str:
    """Returns just the filename (not the path) of the model reference file for the given model reference category.

    Args:
        model_reference_category (MODEL_REFERENCE_CATEGORIES): The category of model reference to get the filename for.

    Returns:
        str: The filename of the model reference file.
    """
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
