"""Constants, especially those to do with paths or network locations, for the horde_model_reference package."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

from horde_model_reference import (
    HordeModelReferenceSettings,
    ai_horde_worker_settings,
    horde_model_reference_settings,
)
from horde_model_reference.meta_consts import (
    MODEL_REFERENCE_CATEGORY,
    github_image_model_reference_categories,
)

PACKAGE_NAME = "horde_model_reference"
"""The name of this package. Also used as the name of the base folder name for all model reference files."""


LEGACY_REFERENCE_FOLDER_NAME: str = "legacy"
"""The default name of the legacy model reference folder.
If you need the default path, use `LEGACY_REFERENCE_FOLDER`."""

DEFAULT_SHOWCASE_FOLDER_NAME: str = "showcase"
"""The default name of the stable diffusion showcase folder. If you need the path, use `SHOWCASE_FOLDER_PATH`."""

META_FOLDER_NAME: str = "meta"
"""The default name of the metadata folder. If you need the path, use the meta_path property."""

META_LEGACY_FOLDER_NAME: str = "legacy"
"""The name of the legacy metadata subfolder within the meta folder."""

META_V2_FOLDER_NAME: str = "v2"
"""The name of the v2 metadata subfolder within the meta folder."""


class HordeModelReferencePaths:
    """A helper class to manage local and remote model reference paths."""

    model_reference_filenames: dict[MODEL_REFERENCE_CATEGORY, str]
    legacy_image_model_github_urls: dict[MODEL_REFERENCE_CATEGORY, str]
    legacy_text_model_github_urls: dict[MODEL_REFERENCE_CATEGORY, str]

    base_path: Path

    @property
    def legacy_path(self) -> Path:
        """Return the path to the legacy model reference folder."""
        return self.base_path.joinpath(LEGACY_REFERENCE_FOLDER_NAME)

    @property
    def showcase_path(self) -> Path:
        """Return the path to the stable diffusion showcase folder."""
        return self.base_path.joinpath(DEFAULT_SHOWCASE_FOLDER_NAME)

    @property
    def meta_path(self) -> Path:
        """Return the path to the metadata folder."""
        return self.base_path.joinpath(META_FOLDER_NAME)

    @property
    def meta_legacy_path(self) -> Path:
        """Return the path to the legacy metadata folder (meta/legacy/)."""
        return self.meta_path.joinpath(META_LEGACY_FOLDER_NAME)

    @property
    def meta_v2_path(self) -> Path:
        """Return the path to the v2 metadata folder (meta/v2/)."""
        return self.meta_path.joinpath(META_V2_FOLDER_NAME)

    log_folder: Path

    _instance: HordeModelReferencePaths | None = None
    _initialized: bool = False

    def __new__(
        cls: type[HordeModelReferencePaths],
        model_reference_settings: HordeModelReferenceSettings,
        cache_home: str | Path,
        log_folder: str | Path | None,
    ) -> HordeModelReferencePaths:
        """Create a singleton instance of HordeModelReferencePaths, if it doesn't already exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls.__init__(
                cls._instance,
                model_reference_settings,
                cache_home,
                log_folder,
            )
        return cls._instance

    def __init__(
        self,
        model_reference_settings: HordeModelReferenceSettings,
        cache_home: str | Path,
        log_folder: str | Path | None,
    ) -> None:
        """Initialize a singleton instance of HordeModelReferencePaths, if it doesn't already exist.

        Args:
            model_reference_settings (HordeModelReferenceSettings): The model reference settings to use.
            cache_home (str | Path): The path to the cache home directory.
            log_folder (str | Path | None): The path to the log folder.
        """
        if HordeModelReferencePaths._initialized:
            return

        HordeModelReferencePaths._initialized = True

        self.model_reference_filenames = {}
        self.legacy_image_model_github_urls = {}
        self.legacy_text_model_github_urls = {}

        self.base_path = Path(cache_home).resolve().joinpath(PACKAGE_NAME)
        self.log_folder = Path(log_folder).resolve() if log_folder else self.base_path.joinpath("logs")

        if model_reference_settings.make_folders:
            logger.info("Making all model reference folders if they don't already exist.")
            logger.info(f"BASE_PATH: {self.base_path}")
            self.make_all_model_reference_folders()

        self.model_reference_filenames[MODEL_REFERENCE_CATEGORY.image_generation] = "stable_diffusion.json"
        self.model_reference_filenames[MODEL_REFERENCE_CATEGORY.text_generation] = "db.json"

        for category in MODEL_REFERENCE_CATEGORY:
            filename: str | None = None
            if category not in self.model_reference_filenames:
                filename = f"{category}.json"
                self.model_reference_filenames[category] = filename
            else:
                filename = self.model_reference_filenames[category]
                logger.trace(
                    f"Using fixed filename for {category}: {filename}",
                )
            composed_url: str | None = None
            if category in github_image_model_reference_categories:
                composed_url = urlparse(
                    horde_model_reference_settings.image_github_repo.compose_full_file_url(filename),
                    allow_fragments=False,
                ).geturl()
                self.legacy_image_model_github_urls[category] = composed_url
            else:
                composed_url = urlparse(
                    horde_model_reference_settings.text_github_repo.compose_full_file_url(filename),
                    allow_fragments=False,
                ).geturl()
                self.legacy_text_model_github_urls[category] = composed_url

            logger.trace(f"Parsed legacy model GitHub URL for {category}: {composed_url}")

        self.model_reference_filenames[MODEL_REFERENCE_CATEGORY.text_generation] = "text_generation.csv"
        logger.trace(
            f"Renaming {MODEL_REFERENCE_CATEGORY.text_generation}: "
            f"{self.model_reference_filenames[MODEL_REFERENCE_CATEGORY.text_generation]}",
        )

    def make_all_model_reference_folders(self) -> None:
        """Create all model reference folders if they don't already exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.log_folder.mkdir(parents=True, exist_ok=True)
        self.legacy_path.mkdir(parents=True, exist_ok=True)
        self.meta_legacy_path.mkdir(parents=True, exist_ok=True)
        self.meta_v2_path.mkdir(parents=True, exist_ok=True)

    def _get_file_name(self, model_reference_category: MODEL_REFERENCE_CATEGORY) -> str:
        if model_reference_category not in self.model_reference_filenames:
            logger.debug(
                f"Generating filename for {model_reference_category}: "
                f"{self.model_reference_filenames[model_reference_category]} "
                "for registered category.",
            )
            self.model_reference_filenames[model_reference_category] = f"{model_reference_category}.json"
        return self.model_reference_filenames[model_reference_category]

    def get_model_reference_filename(
        self,
        model_reference_category: MODEL_REFERENCE_CATEGORY,
        *,
        base_path: str | Path | None = None,
    ) -> str | Path:
        """Return the filename for the given model reference category.

        Args:
            model_reference_category (MODEL_REFERENCE_CATEGORY): The category of model reference to get the filename
                for.
            base_path (str | Path | None): If provided, the base path to the model reference file.
                Defaults to BASE_PATH.

        Returns:
            str: The filename for the given model reference category. If base_path is provided, returns the full path
            from get_model_reference_file_path(...).
        """
        if base_path:
            base_path = Path(base_path)
            return self.get_model_reference_file_path(model_reference_category, base_path=base_path).resolve()

        return self._get_file_name(model_reference_category)

    def get_model_reference_file_path(
        self,
        model_reference_category: MODEL_REFERENCE_CATEGORY,
        *,
        base_path: str | Path | None = None,
    ) -> Path:
        """Return the path to the model reference file for the given model reference category.

        Args:
            model_reference_category (MODEL_REFERENCE_CATEGORY): The category of model reference to get the filename
                for.
            base_path (str | Path): If provided, the base path to the model reference file. Defaults to BASE_PATH.

        Returns:
            path:
        """
        if base_path is None:
            base_path = self.base_path

        base_path = Path(base_path)

        return base_path.joinpath(self._get_file_name(model_reference_category))

    def get_all_model_reference_file_paths(
        self,
        *,
        base_path: str | Path | None = None,
    ) -> dict[MODEL_REFERENCE_CATEGORY, Path | None]:
        """Return the path to the model reference file for the given model reference category.

        Args:
            base_path (str | Path): If provided, the base path to the model reference file. Defaults to BASE_PATH.

        Returns:
            path:
        """
        if base_path is None:
            base_path = self.base_path

        base_path = Path(base_path)

        return_dict: dict[MODEL_REFERENCE_CATEGORY, Path | None] = {}

        for model_reference_category in MODEL_REFERENCE_CATEGORY:
            file_path = self.get_model_reference_file_path(model_reference_category, base_path=base_path)
            if not file_path.exists():
                logger.trace(f"Model reference file does not exist for {model_reference_category}: {file_path}")
                return_dict[model_reference_category] = None
            else:
                return_dict[model_reference_category] = file_path

        return return_dict

    def get_legacy_model_reference_file_path(
        self,
        model_reference_category: MODEL_REFERENCE_CATEGORY,
        *,
        base_path: str | Path | None = None,
    ) -> Path:
        """Return the path to the legacy model reference file for the given model reference category.

        Args:
            model_reference_category (MODEL_REFERENCE_CATEGORY): The category of model reference to get the filename
                for.
            base_path (str | Path): If provided, the base path to the model reference file. Defaults to BASE_PATH.

        Returns:
            path:
        """
        if base_path is None:
            logger.trace("Using default base_path for legacy model reference file path.")
            base_path = self.base_path

        return Path(base_path) / LEGACY_REFERENCE_FOLDER_NAME / self._get_file_name(model_reference_category)

    def get_legacy_metadata_file_path(
        self,
        model_reference_category: MODEL_REFERENCE_CATEGORY,
        *,
        base_path: str | Path | None = None,
    ) -> Path:
        """Return the path to the legacy metadata file for the given model reference category.

        Args:
            model_reference_category: The category of model reference to get the metadata file for.
            base_path: If provided, the base path to the model reference file. Defaults to BASE_PATH.

        Returns:
            Path to legacy metadata file (meta/legacy/{category}_metadata.json)
        """
        if base_path is None:
            base_path = self.base_path

        base_path = Path(base_path)
        return (
            base_path / META_FOLDER_NAME / META_LEGACY_FOLDER_NAME / f"{model_reference_category.value}_metadata.json"
        )

    def get_v2_metadata_file_path(
        self,
        model_reference_category: MODEL_REFERENCE_CATEGORY,
        *,
        base_path: str | Path | None = None,
    ) -> Path:
        """Return the path to the v2 metadata file for the given model reference category.

        Args:
            model_reference_category: The category of model reference to get the metadata file for.
            base_path: If provided, the base path to the model reference file. Defaults to BASE_PATH.

        Returns:
            Path to v2 metadata file (meta/v2/{category}_metadata.json)
        """
        if base_path is None:
            base_path = self.base_path

        base_path = Path(base_path)
        return base_path / META_FOLDER_NAME / META_V2_FOLDER_NAME / f"{model_reference_category.value}_metadata.json"


horde_model_reference_paths = HordeModelReferencePaths(
    model_reference_settings=horde_model_reference_settings,
    cache_home=ai_horde_worker_settings.aiworker_cache_home,
    log_folder=ai_horde_worker_settings.logs_folder,
)
