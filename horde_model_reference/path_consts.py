"""Constants, especially those to do with paths or network locations, for the horde_model_reference package."""

from pathlib import Path

from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORIES

PACKAGE_NAME = "horde_model_reference"

# MODEL_REFERENCE_GITHUB_REPO_OWNER = "db0"
MODEL_REFERENCE_GITHUB_REPO_OWNER = "tazlin"
# MODEL_REFERENCE_GITHUB_REPO_NAME = "horde_model_reference"
MODEL_REFERENCE_GITHUB_REPO_NAME = "AI-Horde-image-model-reference"
# MODEL_REFERENCE_GITHUB_REPO_BRANCH = "main" # "comfy"
MODEL_REFERENCE_GITHUB_REPO_BRANCH = "librarize-modeldb"

# github_repo_url = "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/comfy/"
MODEL_REFERENCE_GITHUB_REPO: str
"""The base URL to the live GitHub repo used to power the horde."""

MODEL_REFERENCE_GITHUB_REPO = f"https://raw.githubusercontent.com/{MODEL_REFERENCE_GITHUB_REPO_OWNER}/{MODEL_REFERENCE_GITHUB_REPO_NAME}/{MODEL_REFERENCE_GITHUB_REPO_BRANCH}/"

LEGACY_MODEL_REFERENCE_GITHUB_REPO: str
"""The base URL to the legacy GitHub repo used to power the horde. This path ends in `horde_model_reference/`."""


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
    *,
    base_path: str | Path = BASE_PATH,
) -> Path:
    """Returns the filename for the given model reference category.

    Args:
        model_reference_category (MODEL_REFERENCE_CATEGORIES): The category of model reference to get the filename for.
        basePath (str | Path): If provided, the base path to the model reference file. Defaults to BASE_PATH.

    Returns:
        path:
    """
    if not isinstance(base_path, Path):
        base_path = Path(base_path)
    return base_path.joinpath(_MODEL_REFERENCE_FILENAMES[model_reference_category])
