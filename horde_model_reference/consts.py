"""Constants, especially those to do with paths, for the horde_model_reference package."""

from pathlib import Path
import enum

# github_repo_url = "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/comfy/"
MODEL_REFERENCE_GITHUB_REPO = (
    "https://raw.githubusercontent.com/tazlin/AI-Horde-image-model-reference/librarize-modeldb/horde_model_reference/"
)
"""The base URL to the live GitHub repo used to power the horde. This path ends in `horde_model_reference/`."""

BASE_PATH = Path(__file__).parent
"""The base path to this folder. Should end in `horde_model_reference/`."""

LEGACY_REFERENCE_FOLDER = BASE_PATH.joinpath("legacy")
"""The path to the legacy model reference folder."""


class MODEL_REFERENCE_TYPE(str, enum.Enum):
    """The types of model reference entries."""

    STABLE_DIFFUSION = "stable_diffusion"
    """A Stable Diffusion model entry."""
    CONTROLNET = "controlnet"


MODEL_REFERENCE_FILENAMES: dict[MODEL_REFERENCE_TYPE, str] = {
    MODEL_REFERENCE_TYPE.STABLE_DIFFUSION: "stable_diffusion.json",
    MODEL_REFERENCE_TYPE.CONTROLNET: "controlnet.json",
}


def get_model_reference_filename(
    model_reference_type: MODEL_REFERENCE_TYPE,
    *,
    basePath: str | Path = BASE_PATH,
) -> Path:
    """Returns the filename for the given model reference type.

    Args:
        model_reference_type (MODEL_REFERENCE_TYPE): The type of model reference to get the filename for.
        basePath (str | Path): If provided, the base path to the model reference file. Defaults to BASE_PATH.

    Returns:
        path:
    """
    if not isinstance(basePath, Path):
        basePath = Path(basePath)
    return basePath.joinpath(MODEL_REFERENCE_FILENAMES[model_reference_type])
