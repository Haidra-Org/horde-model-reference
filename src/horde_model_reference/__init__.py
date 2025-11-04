"""Object models, functions, utilities and service definitions for working with horde ecosystem model references."""

from __future__ import annotations

import urllib.parse
from enum import auto
from typing import Literal

from haidra_core.ai_horde.meta import AIHordeCISettings
from haidra_core.ai_horde.settings import AIHordeWorkerSettings
from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from strenum import StrEnum

SCHEMA_VERSION = "2.0.0"

ai_horde_ci_settings: AIHordeCISettings = AIHordeCISettings()
"""Environment settings for AI Horde CI. See `haidra_core.ai_horde.meta.AIHordeCISettings` for details."""

ai_horde_worker_settings: AIHordeWorkerSettings = AIHordeWorkerSettings()
"""Environment settings for AI Horde workers. See `haidra_core.ai_horde.settings.AIHordeWorkerSettings` for details."""

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
"""Settings for GitHub proxying, if any."""


class GithubRepoSettings(BaseModel):
    """Settings for GitHub integration."""

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

        For example, if the base URL is `https://raw.githubusercontent.com/owner/name/branch/` and the filename is
        `models/model1.json`, the resulting URL will be
        `https://raw.githubusercontent.com/owner/name/branch/models/model1.json`.

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

    @property
    def repo_owner_and_name(self) -> str:
        """Return the GitHub repository in 'owner/name' format."""
        return f"{self.owner}/{self.name}"

    @property
    def git_clone_url(self) -> str:
        """Return the git clone URL for this repository.

        Returns:
            str: The HTTPS git clone URL (e.g., 'https://github.com/owner/name.git').
        """
        return f"https://github.com/{self.owner}/{self.name}.git"


# These `GithubRepoSettings` child classes exist so the generated `.env.example` files are filled in as intended
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


class RedisSettings(BaseModel):
    """Settings for Redis distributed caching in PRIMARY mode."""

    model_config = SettingsConfigDict(
        use_attribute_docstrings=True,
    )

    use_redis: bool = False
    """Whether to use Redis for distributed caching. Only should be used in PRIMARY mode."""

    url: str = "redis://localhost:6379/0"
    """Redis connection URL. Format: redis://[:password]@host:port/db"""

    pool_size: int = 10
    """Connection pool size for Redis connections."""

    socket_timeout: int = 5
    """Socket timeout in seconds for Redis operations."""

    socket_connect_timeout: int = 5
    """Connection timeout in seconds when establishing Redis connection."""

    retry_max_attempts: int = 3
    """Maximum number of retry attempts for failed Redis operations."""

    retry_backoff_seconds: float = 0.5
    """Backoff time in seconds between retry attempts for Redis operations."""

    key_prefix: str = "horde:model_ref"
    """Prefix for all Redis keys to namespace model reference data."""

    ttl_seconds: int | None = None
    """TTL for cached entries in seconds. If None, uses cache_ttl_seconds from main settings."""

    use_pubsub: bool = True
    """Enable pub/sub for cache invalidation across multiple PRIMARY workers."""


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

    image_github_repo: ImageGithubRepoSettings = Field(default_factory=ImageGithubRepoSettings)
    """Settings for the GitHub repository used for image model references."""

    text_github_repo: TextGithubRepoSettings = Field(default_factory=TextGithubRepoSettings)
    """Settings for the GitHub repository used for text model references."""

    def get_repo_by_category(self, category: str) -> GithubRepoSettings:
        """Get the GitHub repository settings for a given model reference category.

        Args:
            category (str): The model reference category (e.g., 'image_generation', 'text_generation').

        Returns:
            GithubRepoSettings: The GitHub repository settings for the specified category.
        """
        if category == MODEL_REFERENCE_CATEGORY.text_generation:
            return self.text_github_repo

        return self.image_github_repo

    cache_ttl_seconds: int = 60
    """The time-to-live for in memory caches of model reference files, in seconds."""

    legacy_download_retry_max_attempts: int = 3
    """The maximum number of attempts to retry downloading a legacy model reference file."""

    legacy_download_retry_backoff_seconds: int = 2
    """The backoff time in seconds between retry attempts when downloading a legacy model reference file."""

    redis: RedisSettings = RedisSettings()
    """Redis settings for distributed caching. Only used in PRIMARY mode for multi-worker deployments."""

    primary_api_url: str | None = "https://stablehorde.net/api/model_references/"
    """URL of PRIMARY server API for REPLICA clients to fetch model references from. \
If None, REPLICA clients will only use GitHub. Example: https://stablehorde.net/api/model_references/"""

    primary_api_timeout: int = 10
    """Timeout in seconds for HTTP requests to PRIMARY API."""

    enable_github_fallback: bool = True
    """Whether REPLICA clients should fallback to GitHub if PRIMARY API is unavailable."""

    github_seed_enabled: bool = False
    """Whether PRIMARY mode should seed from GitHub on first initialization if local files don't exist. \
Only used in PRIMARY mode. If True, will download and convert legacy references once on startup."""

    canonical_format: Literal["legacy", "v2"] = "v2"
    """Which format is the canonical source of truth. Controls which API has write access. \
'v2' (default): v2 API has CRUD, v1 API is read-only (converts from v2 to legacy). \
'legacy': v1 API has CRUD, v2 API is read-only (converts from legacy to v2)."""

    horde_api_timeout: int = 10
    """Timeout in seconds for Horde API requests to fetch model status, statistics, and worker information."""

    horde_api_cache_ttl: int = 60
    """Cache TTL in seconds for Horde API responses. Uses Redis if available, otherwise in-memory caching."""

    statistics_cache_ttl: int = 300
    """Cache TTL in seconds for category statistics. Uses Redis if available, otherwise in-memory caching."""

    audit_cache_ttl: int = 300
    """Cache TTL in seconds for category audit results. Uses Redis if available, otherwise in-memory caching."""

    enable_statistics_precompute: bool = False
    """Enable background pre-computation of statistics. Currently not implemented (future feature)."""

    preferred_file_hosts: list[str] = Field(default_factory=lambda: ["huggingface.co"])
    """Preferred file hosts for deletion risk analysis in audit endpoints."""

    low_usage_threshold_percentage: float = 0.0065
    """Percentage threshold for low usage flag in audit analysis. Default 0.0065% flags bottom ~10% of models. \
Set lower (e.g., 0.005%) to flag fewer models or higher (e.g., 0.01%) to flag more models."""

    @model_validator(mode="after")
    def validate_mode_configuration(self) -> HordeModelReferenceSettings:
        """Validate that settings are appropriate for the configured replication mode."""
        if self.replicate_mode == ReplicateMode.REPLICA and self.redis.use_redis is True:
            logger.warning(
                "Redis settings detected in REPLICA mode. "
                "Redis is only useful in PRIMARY mode for distributed caching across workers. "
                "REPLICA instances should use file-based caching. Redis settings will be ignored."
            )
            self.redis.use_redis = False

        if self.replicate_mode == ReplicateMode.REPLICA and not self.primary_api_url:
            logger.warning(
                "REPLICA mode without primary_api_url configured: "
                "Will only use GitHub for model references (slower, higher bandwidth usage). "
                "Consider setting HORDE_MODEL_REFERENCE_PRIMARY_API_URL to fetch from PRIMARY server."
            )

        if self.replicate_mode == ReplicateMode.PRIMARY and not self.redis.use_redis:
            logger.info(
                "PRIMARY mode without Redis: Single-worker deployment assumed. "
                "For multi-worker PRIMARY deployments, configure Redis for distributed caching "
                "via HORDE_MODEL_REFERENCE_REDIS_URL."
            )

        if self.replicate_mode == ReplicateMode.REPLICA and self.github_seed_enabled:
            logger.warning(
                "github_seed_enabled is set in REPLICA mode. "
                "This setting only applies to PRIMARY mode for initial seeding. "
                "REPLICA instances always fetch from PRIMARY API or GitHub. Setting will be ignored."
            )
            self.github_seed_enabled = False

        if self.canonical_format == "legacy" and self.replicate_mode == ReplicateMode.REPLICA:
            logger.warning(
                "canonical_format='legacy' in REPLICA mode: "
                "v1 API will be read-only. Write operations require PRIMARY mode."
            )

        if self.canonical_format == "legacy" and self.replicate_mode == ReplicateMode.PRIMARY:
            logger.info(
                "canonical_format='legacy' in PRIMARY mode: "
                "v1 API has CRUD operations, v2 API is read-only. "
                "Note: v2 → legacy conversion is not yet implemented."
            )

        return self


horde_model_reference_settings: HordeModelReferenceSettings = HordeModelReferenceSettings()

# Print the github repo settings if they are not the default
if horde_model_reference_settings.image_github_repo != ImageGithubRepoSettings():
    logger.debug("Image GitHub Repo Settings:")
    logger.debug(horde_model_reference_settings.image_github_repo)
    if horde_model_reference_settings.replicate_mode != ReplicateMode.REPLICA:
        logger.debug(f"Replicate mode is set to {horde_model_reference_settings.replicate_mode}.")

if horde_model_reference_settings.text_github_repo != TextGithubRepoSettings():
    logger.debug("Text GitHub Repo Settings:")
    logger.debug(horde_model_reference_settings.text_github_repo)
    if horde_model_reference_settings.replicate_mode != ReplicateMode.REPLICA:
        logger.debug(f"Replicate mode is set to {horde_model_reference_settings.replicate_mode}.")


from .meta_consts import (  # noqa: E402, I001
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

from .model_reference_manager import ModelReferenceManager  # noqa: E402

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
    "ModelReferenceManager",
    "get_model_reference_file_path",
    "get_model_reference_filename",
    "horde_model_reference_paths",
]
