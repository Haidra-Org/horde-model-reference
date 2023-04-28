from pathlib import Path
import enum

# github_repo_url = "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/comfy/"
GITHUB_REPO_BASE_URL = (
    "https://raw.githubusercontent.com/tazlin/AI-Horde-image-model-reference/librarize-modeldb/horde_model_reference/"
)
"""The base URL to the live GitHub used to power the horde. This path ends in `horde_model_reference/`."""

BASE_PATH = Path(__file__).parent
"""The base path to the horde_model_reference folder. """

LEGACY_REFERENCE_FOLDER = BASE_PATH.joinpath("legacy")
"""The path to the legacy model reference folder."""

class ModelReferenceTypes(str, enum.Enum):
    """The types of model reference entries."""
    
    STABLE_DIFFUSION = "stable_diffusion"
    """A Stable Diffusion model entry."""