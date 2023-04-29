"""Constants, especially those to do with paths, for the horde_model_reference package."""

from pathlib import Path
import enum

from horde_model_reference.model_database_records import StableDiffusion_ModelRecord
from horde_model_reference.legacy.legacy_model_database_records import (
    Legacy_Generic_ModelRecord,
    Legacy_StableDiffusion_ModelRecord,
)

# github_repo_url = "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/comfy/"
MODEL_REFERENCE_GITHUB_REPO = (
    "https://raw.githubusercontent.com/tazlin/AI-Horde-image-model-reference/librarize-modeldb/horde_model_reference/"
)
"""The base URL to the live GitHub repo used to power the horde. This path ends in `horde_model_reference/`."""

BASE_PATH = Path(__file__).parent
"""The base path to the image database folder. Should end in `horde_model_reference/`."""

LEGACY_REFERENCE_FOLDER = BASE_PATH.joinpath("legacy")
"""The path to the legacy model reference folder."""


DEFAULT_SHOWCASE_FOLDER_NAME = "showcase"
"""The default name of the showcase folder. If you need the path, use `SHOWCASE_FOLDER_PATH`."""


SHOWCASE_FOLDER_PATH = BASE_PATH.joinpath(DEFAULT_SHOWCASE_FOLDER_NAME)
"""The path to the showcase folder."""


class MODEL_REFERENCE_CATEGORIES(str, enum.Enum):
    """The categories of model reference entries."""

    STABLE_DIFFUSION = "stable_diffusion"
    """A Stable Diffusion model entry."""
    CONTROLNET = "controlnet"


MODEL_REFERENCE_FILENAMES: dict[MODEL_REFERENCE_CATEGORIES, str] = {
    MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION: "stable_diffusion.json",
    MODEL_REFERENCE_CATEGORIES.CONTROLNET: "controlnet.json",
}

MODEL_REFERENCE_RECORD_TYPE_LOOKUP: dict[MODEL_REFERENCE_CATEGORIES, type[StableDiffusion_ModelRecord]] = {
    MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION: StableDiffusion_ModelRecord,  # FIXME
    MODEL_REFERENCE_CATEGORIES.CONTROLNET: StableDiffusion_ModelRecord,  # FIXME
}

MODEL_REFERENCE_LEGACY_TYPE_LOOKUP: dict[MODEL_REFERENCE_CATEGORIES, type[Legacy_Generic_ModelRecord]] = {
    MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION: Legacy_StableDiffusion_ModelRecord,
    MODEL_REFERENCE_CATEGORIES.CONTROLNET: Legacy_Generic_ModelRecord,
}


def get_model_reference_filename(
    model_reference_category: MODEL_REFERENCE_CATEGORIES,
    *,
    basePath: str | Path = BASE_PATH,
) -> Path:
    """Returns the filename for the given model reference category.

    Args:
        model_reference_category (MODEL_REFERENCE_CATEGORIES): The category of model reference to get the filename for.
        basePath (str | Path): If provided, the base path to the model reference file. Defaults to BASE_PATH.

    Returns:
        path:
    """
    if not isinstance(basePath, Path):
        basePath = Path(basePath)
    return basePath.joinpath(MODEL_REFERENCE_FILENAMES[model_reference_category])
