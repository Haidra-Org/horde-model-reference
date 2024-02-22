"""The classes used to convert the legacy model reference to the new format."""

# This whole file is predicated on the idea that the old format suc... wasn't ideal.

# There is a lot of manual wrangling of dicts and failure all around on my part in terms of
# keeping track of what is what and where it is going in a clear, readable way.

# If you can avoid it, don't look at this file. It's not pretty.

import glob
import json
import typing
import urllib.parse
from pathlib import Path

from loguru import logger
from pydantic import ValidationError
from typing_extensions import override

from horde_model_reference import (
    BASE_PATH,
    DEFAULT_SHOWCASE_FOLDER_NAME,
    LEGACY_REFERENCE_FOLDER,
    LOG_FOLDER,
    MODEL_PURPOSE_LOOKUP,
    MODEL_REFERENCE_CATEGORY,
    path_consts,
)
from horde_model_reference.legacy.classes.staging_model_database_records import (
    MODEL_REFERENCE_LEGACY_TYPE_LOOKUP,
    Legacy_StableDiffusion_ModelRecord,
    Staging_StableDiffusion_ModelReference,
    StagingLegacy_Config_DownloadRecord,
    StagingLegacy_Config_FileRecord,
    StagingLegacy_Generic_ModelRecord,
)
from horde_model_reference.meta_consts import MODEL_PURPOSE
from horde_model_reference.model_reference_records import (
    StableDiffusion_ModelRecord,
)
from horde_model_reference.path_consts import (
    GITHUB_REPO_URL,
    PACKAGE_NAME,
)
from horde_model_reference.util import model_name_to_showcase_folder_name


class BaseLegacyConverter:
    """The logic applicable to all legacy model reference converters.
    See normalize_and_convert() for the order of operations critical to the conversion process."""

    legacy_folder_path: Path
    """The folder path to the legacy model reference."""
    legacy_database_path: Path
    """The file path to the legacy stable diffusion model reference database."""
    converted_folder_path: Path
    """The folder path to write write any converted."""
    converted_database_file_path: Path
    """The file path to write the converted stable diffusion model reference database."""

    model_reference_category: MODEL_REFERENCE_CATEGORY
    """The category of model reference to convert."""
    model_reference_type: type[StagingLegacy_Generic_ModelRecord]
    """The `type` (class type) of model reference to convert."""

    all_model_records: dict[str, StagingLegacy_Generic_ModelRecord]
    """All the models entries in found that will be converted."""

    all_validation_errors_log: dict[str, list[str]]
    """All the validation errors that occurred during the conversion. Written to a log file at the end."""

    debug_mode: bool = False
    """If true, include extra information in the error log."""

    log_folder: Path = LOG_FOLDER
    """The folder to write the validation error log to."""

    dry_run: bool = False
    """If true, don't write out the converted database or any log files."""

    def __init__(
        self,
        *,
        legacy_folder_path: str | Path = LEGACY_REFERENCE_FOLDER,
        target_file_folder: str | Path = BASE_PATH,
        log_folder: str | Path = LOG_FOLDER,
        model_reference_category: MODEL_REFERENCE_CATEGORY,
        debug_mode: bool = False,
        dry_run: bool = False,
    ):
        """Initialize an instance of the LegacyConverterBase class.

        Args:
            legacy_folder_path (str | Path, optional): The legacy database folder. Defaults to LEGACY_REFERENCE_FOLDER.
            target_file_folder (str | Path): The folder to write the converted database to.
            model_reference_category (MODEL_REFERENCE_CATEGORY): The category of model reference to convert.
            debug_mode (bool, optional): If true, include extra information in the error log. Defaults to False.
            dry_run (bool, optional): If true, don't write out the converted database or any logs. Defaults to False.
        """
        self.all_model_records = {}
        self.all_validation_errors_log = {}

        self.model_reference_category = model_reference_category
        self.model_reference_type = MODEL_REFERENCE_LEGACY_TYPE_LOOKUP[model_reference_category]

        self.legacy_folder_path = Path(legacy_folder_path)
        self.legacy_database_path = path_consts.get_model_reference_file_path(
            model_reference_category=model_reference_category,
            base_path=legacy_folder_path,
        )
        self.converted_folder_path = Path(target_file_folder)
        self.converted_database_file_path = path_consts.get_model_reference_file_path(
            model_reference_category=model_reference_category,
            base_path=target_file_folder,
        )
        self.debug_mode = debug_mode

        self.log_folder = Path(log_folder)

        self.dry_run = dry_run

    def normalize_and_convert(self) -> bool:
        """Normalizes and converts the legacy model reference database to the new format.

        Returns:
            bool: `True` if the conversion was successful, False otherwise.
        """
        self.pre_parse_records()
        all_model_iterator = self._iterate_over_input_records(self.model_reference_type)
        for model_record_key, model_record_in_progress in all_model_iterator:
            if model_record_in_progress is None:
                raise ValueError(f"CRITICAL: new_record is None! model_record_key = {model_record_key}")

            self.generic_record_sanity_checks(
                model_record_key=model_record_key,
                record=model_record_in_progress,
            )
            self.parse_record(model_record_key=model_record_key, model_record_in_progress=model_record_in_progress)

        self.post_parse_records()
        self.write_out_validation_errors()
        self.write_out_records()

        return True

    def _iterate_over_input_records(
        self,
        model_record_type: type[StagingLegacy_Generic_ModelRecord],
    ) -> typing.Iterator[tuple[str, StagingLegacy_Generic_ModelRecord]]:
        raw_legacy_json_data: dict = {}
        """Return an iterator over the legacy model reference database.

        Yields:
            Iterator[tuple[str, Legacy_Generic_ModelRecord]]: The model record key and the model record.
        """

        with open(self.legacy_database_path) as legacy_model_reference_file:
            raw_legacy_json_data = json.load(legacy_model_reference_file)

        for model_record_key, model_record_contents in raw_legacy_json_data.items():
            try:
                download = self.config_record_pre_parse(model_record_key, model_record_contents)
                model_record_contents["config"]["download"] = download
                if "files" in model_record_contents["config"]:
                    del model_record_contents["config"]["files"]  # New format doesn't have 'files' in the config

                if "showcases" in model_record_contents["config"]:
                    model_record_contents["showcases"] = model_record_contents["config"]["showcases"]
                    del model_record_contents["config"]["showcases"]
                record_as_conversion_class = model_record_type.model_validate(model_record_contents)
                self.all_model_records[model_record_key] = record_as_conversion_class
                yield model_record_key, record_as_conversion_class
            except ValidationError as e:
                error = f"CRITICAL: Error parsing {model_record_key}:\n{e}"
                self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                raise

    def config_record_pre_parse(
        self,
        model_record_key: str,
        model_record_contents: dict,
    ) -> list[StagingLegacy_Config_DownloadRecord]:
        """Parse the config record of the legacy model reference. Changes `model_record_contents`.

        Args:
            model_record_key (str): The key of the model record.
            model_record_contents (dict): The contents of the model record.
        """
        parsed_record_config_files_list: list[StagingLegacy_Config_FileRecord] = []
        parsed_record_config_download_list: list[StagingLegacy_Config_DownloadRecord] = []

        if len(model_record_contents["config"]) > 2:
            error = f"{model_record_key} has more than 2 config entries."
            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

        sha_lookup = {}
        for config_entry in model_record_contents["config"]:
            if config_entry == "files":
                for config_file in model_record_contents["config"][config_entry]:
                    parsed_file_record = StagingLegacy_Config_FileRecord(**config_file)
                    if ".yaml" in parsed_file_record.path:
                        continue

                    # We shift the sha256sum to the download record
                    sha_lookup[parsed_file_record.path] = parsed_file_record.sha256sum
                    parsed_file_record.sha256sum = None

                    parsed_record_config_files_list.append(parsed_file_record)

            elif config_entry == "download":
                for download in model_record_contents["config"][config_entry]:
                    sha_dict = {}
                    if download.get("file_name") and download["file_name"] in sha_lookup:
                        sha_dict = {"sha256sum": sha_lookup[download["file_name"]]}
                    all_params = {**download, **sha_dict}
                    parsed_download_record = StagingLegacy_Config_DownloadRecord(**all_params)

                    if parsed_download_record.sha256sum is None:
                        error = f"{model_record_key} has a download record without a sha256sum."
                        self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                        parsed_download_record.sha256sum = "FIXME"

                    if download.get("file_type") and download["file_type"] == "ckpt":
                        parsed_download_record.file_type = download["file_type"]

                    parsed_record_config_download_list.append(parsed_download_record)

        return parsed_record_config_download_list

    def generic_record_sanity_checks(
        self,
        *,
        model_record_key: str,
        record: StagingLegacy_Generic_ModelRecord,
    ) -> None:
        """Perform sanity checks which apply to all model categories on the given model record."""
        #
        # Non-conformity checks
        #
        if record.name != model_record_key:
            error = f"name mismatch for {model_record_key}."

            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

        if record.available:
            error = f"{model_record_key} is flagged 'available'."

            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

        if record.download_all and self.debug_mode:
            error = f"{model_record_key} has download_all set."

            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

        if record.config is None:
            error = f"{model_record_key} has no config."

            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

        if record.description is None:
            error = f"{model_record_key} has no description."

            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

        if record.style == "":
            error = f"{model_record_key} has no style."

            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

    def pre_parse_records(self) -> None:
        """Override and call super().pre_parse_records() to perform any model category specific pre parsing."""

    def parse_record(
        self,
        model_record_key: str,
        model_record_in_progress: StagingLegacy_Generic_ModelRecord,
    ) -> None:
        """Override and call super().parse_record(..) to perform any model category specific parsing."""

    def post_parse_records(self) -> None:
        """Override and call super().post_parse_records() to perform any model category specific post parsing."""
        for model_record in self.all_model_records.values():
            model_record.purpose = MODEL_PURPOSE_LOOKUP[self.model_reference_category]

    def write_out_records(self) -> None:
        """Write out the parsed records."""
        if self.dry_run:
            return

        with open(self.converted_database_file_path, "w") as new_model_reference_file:
            json.dump(
                self.all_model_records,
                new_model_reference_file,
                indent=4,
                default=lambda o: o.model_dump(
                    exclude_none=True,
                    exclude_unset=True,
                    by_alias=True,
                ),
            )

    def add_validation_error_to_log(
        self,
        *,
        model_record_key: str,
        error: str,
    ) -> None:
        """Add a validation error to the log."""
        if model_record_key not in self.all_validation_errors_log:
            self.all_validation_errors_log[model_record_key] = []
        self.all_validation_errors_log[model_record_key].append(error)

        if self.debug_mode:
            logger.debug(f"{model_record_key} has error: {error}")

    def write_out_validation_errors(self) -> None:
        """Write out the validation errors."""
        if self.dry_run or not self.debug_mode:
            return

        log_file = self.log_folder.joinpath(self.model_reference_category + ".log")
        with open(log_file, "w") as validation_errors_log_file:
            validation_errors_log_file.write(
                json.dumps(
                    self.all_validation_errors_log,
                    indent=4,
                ),
            )


class LegacyStableDiffusionConverter(BaseLegacyConverter):
    showcase_glob_pattern: str = "horde_model_reference/showcase/*"
    """The glob pattern used to find all showcase folders. Defaults to `'horde_model_reference/showcase/*'`."""
    # todo: extract to consts

    all_baseline_categories: dict[str, int]
    """A dictionary of all the baseline types found and the number of times they appear."""
    all_styles: dict[str, int]
    """A dictionary of all the styles found and the number of times they appear."""
    all_tags: dict[str, int]
    """A dictionary of all the tags found and the number of times they appear."""
    all_download_hosts: dict[str, int]
    """A dictionary of all the model hosts found and the number of times they appear."""

    existing_showcase_files: dict[str, list[str]]
    """The pre-existing showcase files found in the target folder."""

    def __init__(
        self,
        *,
        legacy_folder_path: str | Path = LEGACY_REFERENCE_FOLDER,
        target_file_folder: str | Path = BASE_PATH,
        debug_mode: bool = False,
    ):
        """Initialize an instance of the LegacyStableDiffusionConverter class.

        Args:
            legacy_folder_path (str | Path, optional): The legacy database folder. Defaults to LEGACY_REFERENCE_FOLDER.
            target_file_folder (str | Path): The folder to write the converted database to.
            debug_mode (bool, optional): If true, include extra information in the error log. Defaults to False.
        """
        super().__init__(
            legacy_folder_path=legacy_folder_path,
            target_file_folder=target_file_folder,
            model_reference_category=path_consts.MODEL_REFERENCE_CATEGORY.stable_diffusion,
            debug_mode=debug_mode,
        )
        self.all_baseline_categories = {}
        self.all_styles = {}
        self.all_tags = {}
        self.all_download_hosts = {}

    @override
    def pre_parse_records(self) -> None:
        existing_showcase_folders = glob.glob(self.showcase_glob_pattern, recursive=True)
        self.existing_showcase_files = self.get_existing_showcases(existing_showcase_folders)

    @override
    def parse_record(
        self,
        model_record_key: str,
        model_record_in_progress: StagingLegacy_Generic_ModelRecord,
    ) -> None:
        if not isinstance(model_record_in_progress, Legacy_StableDiffusion_ModelRecord):
            raise TypeError(f"Expected {model_record_key} to be a Stable Diffusion record.")
        if model_record_in_progress.style is not None:
            self.all_styles[model_record_in_progress.style] = (
                self.all_styles.get(model_record_in_progress.style, 0) + 1
            )

        if model_record_in_progress.type != "ckpt":
            error = f"{model_record_key} is not a ckpt!"
            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

        #
        # Increment baseline category counter
        #
        model_record_in_progress.baseline = self.convert_legacy_baseline(model_record_in_progress.baseline)
        self.all_baseline_categories[model_record_in_progress.baseline] = (
            self.all_baseline_categories.get(model_record_in_progress.baseline, 0) + 1
        )

        #
        # Showcase handling and sanity checks
        #
        expected_showcase_foldername = model_name_to_showcase_folder_name(model_record_key)
        self.create_showcase_folder(expected_showcase_foldername)

        if model_record_in_progress.showcases is not None and len(model_record_in_progress.showcases) > 0:
            if any("huggingface" in showcase for showcase in model_record_in_progress.showcases):
                error = f"{model_record_key} has a huggingface showcase."
                self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

            if expected_showcase_foldername not in self.existing_showcase_files:
                raise RuntimeError

            if len(self.existing_showcase_files[expected_showcase_foldername]) == 0:
                error = f"{model_record_key} has no showcases defined on disk."
                self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

            if len(model_record_in_progress.showcases) != len(
                self.existing_showcase_files[expected_showcase_foldername],
            ):
                error = (
                    f"{model_record_key} has no showcase folder when it was expected to have one. "
                    "Expected: {expected_showcase_foldername}"
                )
                self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

            model_record_in_progress.showcases = []
            for file in self.existing_showcase_files[expected_showcase_foldername]:
                url_friendly_name = urllib.parse.quote(Path(file).name)
                # if not any(url_friendly_name in showcase for showcase in new_record.showcases):
                #     logger.debug(f"{model_record_key} is missing a showcase for {url_friendly_name}.")
                #     logger.debug(f"{new_record.showcases=}")
                #     continue
                expected_github_location = urllib.parse.urljoin(
                    GITHUB_REPO_URL,
                    f"{PACKAGE_NAME}/{DEFAULT_SHOWCASE_FOLDER_NAME}/{expected_showcase_foldername}/{url_friendly_name}",
                )
                model_record_in_progress.showcases.append(expected_github_location)
        #
        # Increment tag counter
        #
        if model_record_in_progress.tags is not None:
            for tag in model_record_in_progress.tags:
                self.all_tags[tag] = self.all_tags.get(tag, 0) + 1

        #
        # Config handling and sanity checks
        #
        if len(model_record_in_progress.config) == 0:
            error = f"{model_record_key} has no config."
            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

        config_entries = model_record_in_progress.config
        found_hosts = self.normalize_and_convert_config_entries(
            model_record_key=model_record_key,
            config_entries=config_entries,
        )

        #
        # Increment host counter
        #
        for found_host in found_hosts:
            self.all_download_hosts[found_host] = self.all_download_hosts.get(found_host, 0) + 1

    @override
    def post_parse_records(self) -> None:
        super().post_parse_records()
        final_on_disk_showcase_folders = glob.glob(self.showcase_glob_pattern, recursive=True)
        for folder in final_on_disk_showcase_folders:
            parsed_folder = Path(folder)

            if parsed_folder.is_file():
                continue

            if not any(parsed_folder.iterdir()):
                error = f"showcase folder '{parsed_folder.name}' is empty."
                self.add_validation_error_to_log(model_record_key=parsed_folder.name, error=error)

        final_on_disk_showcase_folders_names = [
            Path(folder).name for folder in final_on_disk_showcase_folders if Path(folder).is_dir()
        ]
        final_expected_showcase_folders = [
            model_name_to_showcase_folder_name(model_name) for model_name in self.all_model_records
        ]

        for folder in final_on_disk_showcase_folders_names:
            if folder not in final_expected_showcase_folders:
                error = f"folder '{folder}' is not in the model records."
                self.add_validation_error_to_log(model_record_key=folder, error=error)

        if self.debug_mode:
            logger.debug(f"{self.all_styles=}")
            logger.debug(f"{self.all_baseline_categories=}")
            logger.debug(f"{self.all_tags=}")
            logger.debug(f"{self.all_download_hosts=}")
            logger.info(f"Total number of models: {len(self.all_model_records)}")
            logger.info(f"Total number of showcase folders: {len(final_on_disk_showcase_folders_names)}")
            logger.info(f"Total number of models with validation issues: {len(self.all_validation_errors_log)}")

    @override
    def write_out_records(self) -> None:
        sanity_check: dict[str, Legacy_StableDiffusion_ModelRecord] = {
            key: value
            for key, value in self.all_model_records.items()
            if isinstance(value, Legacy_StableDiffusion_ModelRecord)
        }
        if len(sanity_check) != len(self.all_model_records):
            raise ValueError("CRITICAL: Not all records are of the correct type.")

        Staging_StableDiffusion_ModelReference(
            baseline=self.all_baseline_categories,
            styles=self.all_styles,
            tags=self.all_tags,
            download_hosts=self.all_download_hosts,
            models=sanity_check,
        )

        models_in_doc_root = {
            k: v.model_dump(
                exclude_none=True,
                exclude_unset=True,
                by_alias=True,
            )
            for k, v in self.all_model_records.items()
        }

        try:
            # If this fails, we have a problem. By definition, the model reference should be converted by this point
            # and ready to be cast to the new model reference type.
            for model_entry in sanity_check.values():
                model_entry_as_dict = model_entry.model_dump(by_alias=True)
                model_entry_as_dict["purpose"] = MODEL_PURPOSE.image_generation
                StableDiffusion_ModelRecord(**model_entry_as_dict)
        except ValidationError as e:
            logger.exception(e)
            logger.exception("CRITICAL: Failed to convert to new model reference type.")
            raise

        if self.dry_run:
            return

        with open(self.converted_database_file_path, "w") as testfile:
            testfile.write(
                json.dumps(
                    models_in_doc_root,
                    indent=4,
                )
                + "\n",
            )

        logger.debug(f"Converted database written to: {self.converted_database_file_path}")

    def get_existing_showcases(
        self,
        existing_showcase_folders: list[str],
    ) -> dict[str, list[str]]:
        """Return a dictionary of existing showcase files, keyed by the showcase folder name.

        Args:
            existing_showcase_folders (list[str]): The list of existing showcase folders.

        Returns:
            dict[str, list[str]]: A dictionary of existing showcase files, keyed by the showcase folder name.
        """
        existing_showcase_files: dict[str, list[str]] = {}
        for showcase_folder in existing_showcase_folders:
            model_showcase_files = glob.glob(str(Path(showcase_folder).joinpath("*")), recursive=True)
            model_showcase_folder_name = model_name_to_showcase_folder_name(Path(showcase_folder).name)

            existing_showcase_files[model_showcase_folder_name] = model_showcase_files

        return existing_showcase_files

    def convert_legacy_baseline(self, baseline: str):
        """Returns the new standardized baseline name for the given legacy baseline name."""
        if baseline == "stable diffusion 1":
            baseline = "stable_diffusion_1"
            # new_record.baseline_trained_resolution = 256
        elif baseline == "stable diffusion 2":
            baseline = "stable_diffusion_2_768"
        elif baseline == "stable diffusion 2 512":
            baseline = "stable_diffusion_2_512"
        elif baseline == "stable_diffusion_xl":
            baseline = "stable_diffusion_xl"
        elif baseline == "stable_cascade":
            baseline = "stable_cascade"
        return baseline

    def create_showcase_folder(self, showcase_foldername: str) -> None:
        """Create a showcase folder with the given name.

        Args:
            showcase_foldername (str): The name of the showcase folder to create.
        """

        if showcase_foldername not in self.existing_showcase_files:
            self.existing_showcase_files[showcase_foldername] = []

        newFolder = self.converted_folder_path.joinpath(path_consts.DEFAULT_SHOWCASE_FOLDER_NAME)
        newFolder = newFolder.joinpath(showcase_foldername)
        newFolder.mkdir(parents=True, exist_ok=True)

    def normalize_and_convert_config_entries(
        self,
        *,
        model_record_key: str,
        config_entries: dict[str, list[StagingLegacy_Config_FileRecord | StagingLegacy_Config_DownloadRecord]],
    ) -> dict[str, int]:
        """Normalize and convert the config entries. This changes the contents of param `config_entries`.

        Args:
            model_record_key (str): The key of the model record.
            config_entries (see type hints): The config entries to normalize and convert.

        Raises:
            TypeError: Raised if a config file definition is under the wrong key.

        Returns:
            dict[str, int]: A dict of the hosts and the number of files they host for this model.
        """
        download_hosts: dict[str, int] = {}
        for config_entry_key, config_entry_object in config_entries.items():
            if config_entry_key == "files":
                for config_file in config_entry_object:
                    if not isinstance(config_file, StagingLegacy_Config_FileRecord):
                        logger.exception(f"{model_record_key} is in 'files' but isn't a `Legacy_Config_FileRecord`!")
                        raise TypeError("Expected `Legacy_Config_FileRecord`.")
                    if config_file.path is None or config_file.path == "":
                        error = f"{model_record_key} has a config file with no path."
                        self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

                    if ".yaml" in config_file.path:
                        if config_file.path != "v2-inference-v.yaml" and config_file.path != "v1-inference.yaml":
                            error = f"{model_record_key} has a non-standard config."
                            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                        continue

                    if ".ckpt" not in config_file.path:
                        error = f"{model_record_key} has a config file with an invalid path."
                        self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

                    if config_file.sha256sum is None or config_file.sha256sum == "":
                        error = f"{model_record_key} has a config file with no sha256sum."
                        self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                    else:
                        if len(config_file.sha256sum) != 64:
                            error = f"{model_record_key} has a config file with an invalid sha256sum."
                            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

            elif config_entry_key == "download":
                for download in config_entry_object:
                    if not isinstance(download, StagingLegacy_Config_DownloadRecord):
                        logger.exception(
                            f"{model_record_key} is in 'download' but isn't a `Legacy_Config_DownloadRecord`!",
                        )
                        raise TypeError("Expected `Legacy_Config_DownloadRecord`.")
                    if download.file_name is None or download.file_name == "":
                        error = f"{model_record_key} has a download with no file_name."
                        self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

                    if download.file_path is None or download.file_path != "":
                        error = f"{model_record_key} has a download with a file_path."
                        self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

                    if download.file_url is None or download.file_url == "":
                        error = f"{model_record_key} has a download with no file_url."
                        self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                        continue

                    if "civitai" in download.file_url:
                        download.known_slow_download = True

                    try:
                        host = urllib.parse.urlparse(download.file_url).netloc
                        download_hosts[host] = download_hosts.get(host, 0) + 1
                    except Exception:
                        error = f"{model_record_key} has a download with an invalid file_url."
                        self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                        raise

        return download_hosts


class LegacyClipConverter(BaseLegacyConverter):
    def __init__(
        self,
        *,
        legacy_folder_path: str | Path = LEGACY_REFERENCE_FOLDER,
        target_file_folder: str | Path = BASE_PATH,
        debug_mode: bool = False,
    ):
        super().__init__(
            legacy_folder_path=legacy_folder_path,
            target_file_folder=target_file_folder,
            model_reference_category=MODEL_REFERENCE_CATEGORY.clip,
            debug_mode=debug_mode,
        )

    @override
    def config_record_pre_parse(
        self,
        model_record_key: str,
        model_record_contents: dict,
    ) -> list[StagingLegacy_Config_DownloadRecord]:
        new_record_config_download_list: list[StagingLegacy_Config_DownloadRecord] = []
        if len(model_record_contents["config"]) > 2:
            error = f"{model_record_key} has more than 2 config entries."
            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
        for config_entry in model_record_contents["config"]:
            if config_entry == "files":
                continue
            if config_entry == "download":
                for download in model_record_contents["config"][config_entry]:
                    # Skip if file_url is missing
                    if download.get("file_url") is None or download.get("file_url") == "":
                        continue
                    parsed_download_record = StagingLegacy_Config_DownloadRecord(**download)
                    parsed_download_record.file_name = model_record_key.replace("/", "-") + ".pt"
                    parsed_download_record.sha256sum = "FIXME"
                    error = f"{model_record_key} has no sha256sum."
                    self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                    new_record_config_download_list.append(parsed_download_record)

        return new_record_config_download_list
