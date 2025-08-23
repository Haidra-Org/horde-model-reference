"""Object models, functions, utilities and service definitions for working with horde ecosystem model references."""

from __future__ import annotations

import urllib.parse
from enum import auto

from haidra_core.ai_horde.meta import AIHordeCISettings
from haidra_core.ai_horde.settings import AIHordeWorkerSettings
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from strenum import StrEnum

ai_horde_ci_settings: AIHordeCISettings = AIHordeCISettings()
ai_horde_worker_settings: AIHordeWorkerSettings = AIHordeWorkerSettings()

GITHUB_REPO_OWNER = "Haidra-Org"
GITHUB_IMAGE_REPO_NAME = "AI-Horde-image-model-reference"
GITHUB_TEXT_REPO_NAME = "AI-Horde-text-model-reference"
GITHUB_REPO_BRANCH = "main"


class GithubProxySettings(BaseSettings):
    """Settings for GitHub proxying."""

    model_config = SettingsConfigDict(
        use_attribute_docstrings=True,
    )

    github_proxy_url_base: str | None = None
    """The base URL for a http(s) GitHub proxy. If None, no proxy is used. This is intended for users where github\
        may be blocked."""


github_proxy_settings = GithubProxySettings()


class GithubRepoSettings(BaseModel):
    """Settings for GitHub integration."""

    # TODO: HORDE_PROXY_URL_BASE

    model_config = SettingsConfigDict(
        use_attribute_docstrings=True,
    )

    owner: str = GITHUB_REPO_OWNER
    """The GitHub owner of the repository."""

    name: str
    """The GitHub name of the repository."""

    branch: str = GITHUB_REPO_BRANCH
    """The GitHub branch of the repository."""

    proxy_settings: GithubProxySettings | None = Field(
        default=None,
        exclude=True,
    )
    """Settings for the GitHub proxy, if any."""

    _url_format_string: str = PrivateAttr(default="https://raw.githubusercontent.com/{owner}/{name}/{branch}/")

    @model_validator(mode="after")
    def check_resulting_url_valid(self) -> GithubRepoSettings:
        urllib.parse.urlparse(self.url_base_only, allow_fragments=False)
        return self

    @property
    def url_base_only(self) -> str:
        """Return the base URL for the GitHub repository, without any specific file."""
        return self._url_format_string.format(
            owner=self.owner,
            name=self.name,
            branch=self.branch,
        )

    def compose_full_file_url(self, filename: str) -> str:
        """Compose the full URL to a file in the repository.

        Args:
            filename (str): The filename to compose the URL for.

        Returns:
            str: The full URL to the file.
        """
        full_url = urllib.parse.urljoin(self.url_base_only + "/", filename)
        if self.proxy_settings and self.proxy_settings.github_proxy_url_base:
            proxied_url = urllib.parse.urljoin(self.proxy_settings.github_proxy_url_base + "/", full_url)
            logger.debug(f"Using proxied URL for {filename}: {proxied_url}")
            return proxied_url

        return full_url


# These child classes exist so the generated `.env.example` files are filled in as intended
# They have no practical purpose beyond that.
class TextGithubRepoSettings(GithubRepoSettings):
    """Settings for the GitHub repository used for text model references."""

    name: str = GITHUB_TEXT_REPO_NAME
    """The name of the GitHub repository used for text model references."""


class ImageGithubRepoSettings(GithubRepoSettings):
    """Settings for the GitHub repository used for image model references."""

    name: str = GITHUB_IMAGE_REPO_NAME
    """The name of the GitHub repository used for image model references."""


class ReplicateMode(StrEnum):
    """Indicates if copies of the model reference are canonical or replicated."""

    PRIMARY = auto()
    """The model references are the primary (canonical) copies and so changes will replicate to clients."""
    REPLICA = auto()
    """The model references are replicas (non-canonical copies). Changes are not tracked and may be lost."""


class HordeModelReferenceSettings(BaseSettings):
    """Settings for the Horde Model Reference package."""

    model_config = SettingsConfigDict(
        env_prefix="HORDE_MODEL_REFERENCE_",
        env_nested_delimiter="_",
        nested_model_default_partial_update=True,
        use_attribute_docstrings=True,
    )

    replicate_mode: ReplicateMode = ReplicateMode.REPLICA
    """Indicates if copies of the model reference are canonical or replicated. Clients should always be replicas."""

    make_folders: bool = False
    """Whether to create the default model reference folders on initialization."""

    image_github_repo: ImageGithubRepoSettings = ImageGithubRepoSettings()
    """Settings for the GitHub repository used for image model references."""

    text_github_repo: TextGithubRepoSettings = TextGithubRepoSettings()
    """Settings for the GitHub repository used for text model references."""

    cache_ttl_seconds: int = 60
    """The time-to-live for in memory caches of model reference files, in seconds."""


horde_model_reference_settings: HordeModelReferenceSettings = HordeModelReferenceSettings()

# Print the github repo settings if they are not the default
if horde_model_reference_settings.image_github_repo != ImageGithubRepoSettings():
    logger.info("Image GitHub Repo Settings:")
    logger.info(horde_model_reference_settings.image_github_repo)

if horde_model_reference_settings.text_github_repo != TextGithubRepoSettings():
    logger.info("Text GitHub Repo Settings:")
    logger.info(horde_model_reference_settings.text_github_repo)


from .meta_consts import (  # noqa: E402
    KNOWN_IMAGE_GENERATION_BASELINE,
    KNOWN_TAGS,
    MODEL_CLASSIFICATION_LOOKUP,
    MODEL_DOMAIN,
    MODEL_PURPOSE,
    MODEL_REFERENCE_CATEGORY,
    MODEL_STYLE,
    ModelClassification,
)
from .path_consts import (  # noqa: E402
    DEFAULT_SHOWCASE_FOLDER_NAME,
    horde_model_reference_paths,
)

__all__ = [
    "BASE_PATH",
    "DEFAULT_SHOWCASE_FOLDER_NAME",
    "KNOWN_IMAGE_GENERATION_BASELINE",
    "KNOWN_TAGS",
    "MODEL_CLASSIFICATION_LOOKUP",
    "MODEL_DOMAIN",
    "MODEL_PURPOSE",
    "MODEL_REFERENCE_CATEGORY",
    "MODEL_STYLE",
    "ModelClassification",
    "get_model_reference_file_path",
    "get_model_reference_filename",
    "horde_model_reference_paths",
]
