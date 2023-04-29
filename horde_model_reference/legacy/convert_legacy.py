import glob
import json
import typing
import urllib.parse
from pathlib import Path

from pydantic import ValidationError
from typing_extensions import override

from horde_model_reference import consts
from horde_model_reference.consts import (
    BASE_PATH,
    LEGACY_REFERENCE_FOLDER,
    MODEL_REFERENCE_CATEGORIES,
    MODEL_REFERENCE_GITHUB_REPO,
)
from horde_model_reference.legacy.legacy_model_database_records import (
    MODEL_REFERENCE_LEGACY_TYPE_LOOKUP,
    Legacy_Config_DownloadRecord,
    Legacy_Config_FileRecord,
    Legacy_Generic_ModelRecord,
    Legacy_Generic_ModelReference,
    Legacy_StableDiffusion_ModelRecord,
    Legacy_StableDiffusion_ModelReference,
)
from horde_model_reference.model_database_records import (
    MODEL_PURPOSE_LOOKUP,
    MODEL_REFERENCE_TYPE_LOOKUP,
    Generic_ModelReference,
    StableDiffusion_ModelReference,
)
from horde_model_reference.util import model_name_to_showcase_folder_name


class BaseLegacyConverter:
    legacy_folder_path: Path
    """The folder path to the legacy model reference."""
    legacy_database_path: Path
    """The file path to the legacy stable diffusion model reference database."""
    converted_folder_path: Path
    """The folder path to write write any converted."""
    converted_database_file_path: Path
    """The file path to write the converted stable diffusion model reference database."""

    model_reference_category: MODEL_REFERENCE_CATEGORIES
    model_reference_type: type[Legacy_Generic_ModelRecord]

    all_model_records: dict[str, Legacy_Generic_ModelRecord]
    """All the models entries in found that will be converted."""

    all_validation_errors_log: dict[str, list[str]]

    debug_mode: bool = False
    print_errors: bool = True

    def __init__(
        self,
        *,
        legacy_folder_path: str | Path = LEGACY_REFERENCE_FOLDER,
        target_file_folder: str | Path = BASE_PATH,
        model_reference_category: MODEL_REFERENCE_CATEGORIES,
        print_errors: bool = True,
        debug_mode: bool = False,
    ):
        """Initialize an instance of the LegacyConverterBase class.

        Args:
            legacy_folder_path (str | Path, optional): The legacy database folder. Defaults to LEGACY_REFERENCE_FOLDER.
            target_file_folder (str | Path): The folder to write the converted database to.
            model_reference_category (MODEL_REFERENCE_CATEGORIES): The category of model reference to convert.
            print_errors (bool, optional): Whether to print errors in the conversion to `stdout`. Defaults to True.
            debug_mode (bool, optional): If true, include extra information in the error log. Defaults to False.
        """
        self.all_model_records = {}
        self.all_validation_errors_log = {}

        self.model_reference_category = model_reference_category
        self.model_reference_type = MODEL_REFERENCE_LEGACY_TYPE_LOOKUP[model_reference_category]

        self.legacy_folder_path = Path(legacy_folder_path)
        self.legacy_database_path = consts.get_model_reference_filename(
            model_reference_category=model_reference_category,
            basePath=legacy_folder_path,
        )
        self.converted_folder_path = Path(target_file_folder)
        self.converted_database_file_path = consts.get_model_reference_filename(
            model_reference_category=model_reference_category,
            basePath=target_file_folder,
        )
        self.debug_mode = debug_mode
        self.print_errors = print_errors

    def normalize_and_convert(self) -> bool:
        """Normalizes and converts the legacy model reference database to the new format.

        Returns:
            bool: True if the conversion was successful, False otherwise.
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
        self.write_out_records()

        return True

    def _iterate_over_input_records(
        self,
        modelrecord_type: type[Legacy_Generic_ModelRecord],
    ) -> typing.Iterator[tuple[str, Legacy_Generic_ModelRecord]]:
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
                model_record_contents["config"]["files"] = []
                record_as_conversion_class = modelrecord_type(**model_record_contents)
                self.all_model_records[model_record_key] = record_as_conversion_class
                yield model_record_key, record_as_conversion_class
            except ValidationError as e:
                error = f"CRITICAL: Error parsing {model_record_key}:\n{e}"
                self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                raise e

    def config_record_pre_parse(
        self,
        model_record_key: str,
        model_record_contents: dict,
    ) -> list[Legacy_Config_DownloadRecord]:
        """Parse the config record of the legacy model reference. Changes `model_record_contents`.

        Args:
            model_record_key (str): The key of the model record.
            model_record_contents (dict): The contents of the model record.
        """
        new_record_config_files_list: list[Legacy_Config_FileRecord] = []
        new_record_config_download_list: list[Legacy_Config_DownloadRecord] = []
        if len(model_record_contents["config"]) > 2:
            error = f"{model_record_key} has more than 2 config entries."
            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
        sha_lookup = {}
        for config_entry in model_record_contents["config"]:
            if config_entry == "files":
                for config_file in model_record_contents["config"][config_entry]:
                    parsed_file_record = Legacy_Config_FileRecord(**config_file)
                    if ".yaml" in parsed_file_record.path:
                        continue
                    sha_lookup[parsed_file_record.path] = parsed_file_record.sha256sum
                    parsed_file_record.sha256sum = None
                    new_record_config_files_list.append(parsed_file_record)
            elif config_entry == "download":
                for download in model_record_contents["config"][config_entry]:
                    parsed_download_record = Legacy_Config_DownloadRecord(**download)
                    parsed_download_record.sha256sum = sha_lookup[parsed_download_record.file_name]
                    new_record_config_download_list.append(parsed_download_record)

        return new_record_config_download_list

    def generic_record_sanity_checks(
        self,
        *,
        model_record_key: str,
        record: Legacy_Generic_ModelRecord,
    ) -> None:
        #
        # Non-conformity checks
        #
        if record.name != model_record_key:
            error = f"name mismatch for {model_record_key}."

            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

        if record.available:
            error = f"{model_record_key} is flagged 'available'."

            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

        if record.download_all:
            if self.debug_mode:
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
        """Perform any pre parsing tasks."""
        pass

    def parse_record(
        self,
        model_record_key: str,
        model_record_in_progress: Legacy_Generic_ModelRecord,
    ) -> None:
        """Override and call super().parse_record(..) to perform any model category specific parsing."""
        pass

    def post_parse_records(self) -> None:
        """Perform any post parsing tasks."""
        for model_record in self.all_model_records.values():
            model_record.model_purpose = MODEL_PURPOSE_LOOKUP[self.model_reference_category]
        pass

    def write_out_records(self) -> None:
        """Write out the parsed records."""
        new_reference = None
        type_to_convert_to = MODEL_REFERENCE_TYPE_LOOKUP[self.model_reference_category]
        try:
            _ = Legacy_Generic_ModelReference(models=self.all_model_records)
            new_reference = type_to_convert_to(models=_.models)
            pass
        except ValidationError as e:
            print(f"CRITICAL: Failed to convert to new model reference type {type_to_convert_to}.")
            raise e

        with open(self.converted_database_file_path, "w") as new_model_reference_file:
            new_model_reference_file.write(
                new_reference.json(
                    indent=4,
                    exclude_defaults=True,
                    exclude_none=True,
                    exclude_unset=True,
                )
            )

    def add_validation_error_to_log(
        self,
        *,
        model_record_key: str,
        error: str,
    ) -> None:
        if model_record_key not in self.all_validation_errors_log:
            self.all_validation_errors_log[model_record_key] = []
        self.all_validation_errors_log[model_record_key].append(error)
        if self.print_errors:
            print("-> " + error)


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
    all_model_hosts: dict[str, int]
    """A dictionary of all the model hosts found and the number of times they appear."""

    existing_showcase_files: dict[str, list[str]]
    """The pre-existing showcase files found in the target folder."""

    def __init__(
        self,
        *,
        legacy_folder_path: str | Path = LEGACY_REFERENCE_FOLDER,
        target_file_folder: str | Path = BASE_PATH,
        print_errors: bool = True,
        debug_mode: bool = False,
    ):
        """Initialize an instance of the LegacyStableDiffusionConverter class.

        Args:
            legacy_folder_path (str | Path, optional): The legacy database folder. Defaults to LEGACY_REFERENCE_FOLDER.
            target_file_folder (str | Path): The folder to write the converted database to.
            print_errors (bool, optional): Whether to print errors in the conversion to `stdout`. Defaults to True.
            debug_mode (bool, optional): If true, include extra information in the error log. Defaults to False.
        """
        super().__init__(
            legacy_folder_path=legacy_folder_path,
            target_file_folder=target_file_folder,
            model_reference_category=consts.MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION,
            debug_mode=debug_mode,
            print_errors=print_errors,
        )
        self.all_baseline_categories = {}
        self.all_styles = {}
        self.all_tags = {}
        self.all_model_hosts = {}

    @override
    def pre_parse_records(self) -> None:
        existing_showcase_folders = glob.glob(self.showcase_glob_pattern, recursive=True)
        self.existing_showcase_files = self.get_existing_showcases(existing_showcase_folders)

    @override
    def parse_record(
        self,
        model_record_key: str,
        model_record_in_progress: Legacy_Generic_ModelRecord,
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
                error = f"{model_record_key} has no showcase folder. Expected: {expected_showcase_foldername}"
                self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

            model_record_in_progress.showcases = []
            for file in self.existing_showcase_files[expected_showcase_foldername]:
                url_friendly_name = urllib.parse.quote(Path(file).name)
                # if not any(url_friendly_name in showcase for showcase in new_record.showcases):
                #     print(f"{model_record_key} is missing a showcase for {url_friendly_name}.")
                #     print(f"{new_record.showcases=}")
                #     continue
                expected_github_location = urllib.parse.urljoin(
                    MODEL_REFERENCE_GITHUB_REPO,
                    f"{consts.DEFAULT_SHOWCASE_FOLDER_NAME}/{expected_showcase_foldername}/{url_friendly_name}",
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
            self.all_model_hosts[found_host] = self.all_model_hosts.get(found_host, 0) + 1

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

        print()
        print(f"{self.all_styles=}")
        print(f"{self.all_baseline_categories=}")
        print(f"{self.all_tags=}")
        print(f"{self.all_model_hosts=}")

        print()
        print(f"Total number of models: {len(self.all_model_records)}")
        print(f"Total number of showcase folders: {len(final_on_disk_showcase_folders_names)}")

        print()
        print(f"Total number of models with errors: {len(self.all_validation_errors_log)}")
        print()
        print("Errors and warnings are listed above on lines prefixed with `-> `")

    @override
    def write_out_records(self) -> None:
        sanity_check: dict[str, Legacy_StableDiffusion_ModelRecord] = {
            key: value
            for key, value in self.all_model_records.items()
            if isinstance(value, Legacy_StableDiffusion_ModelRecord)
        }
        if len(sanity_check) != len(self.all_model_records):
            raise ValueError("CRITICAL: Not all records are of the correct type.")

        modelReference = Legacy_StableDiffusion_ModelReference(
            baseline_categories=self.all_baseline_categories,
            styles=self.all_styles,
            tags=self.all_tags,
            model_hosts=self.all_model_hosts,
            models=sanity_check,
        )
        jsonToWrite = modelReference.json(
            indent=4,
            exclude_defaults=True,
            exclude_none=True,
            exclude_unset=True,
            exclude={"nsfw"},
        )

        try:
            # If this fails, we have a problem. By definition, the model reference should be converted by this point
            # and ready to be cast to the new model reference type.
            StableDiffusion_ModelReference(**json.loads(jsonToWrite))
        except ValidationError as e:
            print(e)
            print("CRITICAL: Failed to convert to new model reference type.")
            raise e

        with open(self.converted_database_file_path, "w") as testfile:
            testfile.write(jsonToWrite)

        print("Converted database passes validation and was written to disk successfully.")
        print(f"Converted database written to: {self.converted_database_file_path}")

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
        return baseline

    def create_showcase_folder(self, showcase_foldername: str) -> None:
        """Create a showcase folder with the given name.

        Args:
            showcase_foldername (str): The name of the showcase folder to create.
        """
        newFolder = self.converted_folder_path.joinpath(consts.DEFAULT_SHOWCASE_FOLDER_NAME)
        newFolder = newFolder.joinpath(showcase_foldername)
        newFolder.mkdir(parents=True, exist_ok=True)

    def normalize_and_convert_config_entries(
        self,
        *,
        model_record_key: str,
        config_entries: dict[str, list[Legacy_Config_FileRecord | Legacy_Config_DownloadRecord]],
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
                    if not isinstance(config_file, Legacy_Config_FileRecord):
                        print(f"{model_record_key} is in 'files' but isn't a `Legacy_Config_FileRecord`!")
                        raise TypeError("Expected `Legacy_Config_FileRecord`.")
                    if config_file.path is None or config_file.path == "":
                        error = f"{model_record_key} has a config file with no path."
                        self.add_validation_error_to_log(model_record_key=model_record_key, error=error)

                    if ".yaml" in config_file.path:
                        if config_file.path != "v2-inference-v.yaml" and config_file.path != "v1-inference.yaml":
                            error = f"{model_record_key} has a non-standard config."
                            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                        continue
                    elif ".ckpt" not in config_file.path:
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
                    if not isinstance(download, Legacy_Config_DownloadRecord):
                        print(f"{model_record_key} is in 'download' but isn't a `Legacy_Config_DownloadRecord`!")
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
                    except Exception as e:
                        error = f"{model_record_key} has a download with an invalid file_url."
                        self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                        raise e

        return download_hosts


class LegacyClipConverter(BaseLegacyConverter):
    def __init__(
        self,
        *,
        legacy_folder_path: str | Path = LEGACY_REFERENCE_FOLDER,
        target_file_folder: str | Path = BASE_PATH,
        print_errors: bool = True,
        debug_mode: bool = False,
    ):
        super().__init__(
            legacy_folder_path=legacy_folder_path,
            target_file_folder=target_file_folder,
            model_reference_category=MODEL_REFERENCE_CATEGORIES.CLIP,
            print_errors=print_errors,
            debug_mode=debug_mode,
        )

    @override
    def config_record_pre_parse(
        self,
        model_record_key: str,
        model_record_contents: dict,
    ) -> list[Legacy_Config_DownloadRecord]:
        new_record_config_download_list: list[Legacy_Config_DownloadRecord] = []
        if len(model_record_contents["config"]) > 2:
            error = f"{model_record_key} has more than 2 config entries."
            self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
        for config_entry in model_record_contents["config"]:
            if config_entry == "files":
                continue
            elif config_entry == "download":
                for download in model_record_contents["config"][config_entry]:
                    # Skip if file_url is missing
                    if download.get("file_url") is None or download.get("file_url") == "":
                        continue
                    parsed_download_record = Legacy_Config_DownloadRecord(**download)
                    parsed_download_record.file_name = model_record_key.replace("/", "-") + ".pt"
                    parsed_download_record.sha256sum = "FIXME"
                    error = f"{model_record_key} has no sha256sum."
                    self.add_validation_error_to_log(model_record_key=model_record_key, error=error)
                    new_record_config_download_list.append(parsed_download_record)

        return new_record_config_download_list


if __name__ == "__main__":
    sd_converter = LegacyStableDiffusionConverter(
        legacy_folder_path=Path(__file__).parent,
        target_file_folder=Path(__file__).parent.parent,
        debug_mode=True,
        print_errors=True,
    )
    # sd_converter.normalize_and_convert()

    clip_converter = LegacyClipConverter(
        legacy_folder_path=Path(__file__).parent,
        target_file_folder=Path(__file__).parent.parent,
        debug_mode=True,
        print_errors=True,
    )
    clip_converter.normalize_and_convert()

    non_stablediffusion = [
        x for x in consts.MODEL_REFERENCE_CATEGORIES if x != consts.MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION
    ]

    nor_clip = [x for x in non_stablediffusion if x != consts.MODEL_REFERENCE_CATEGORIES.CLIP]

    for model_category in nor_clip:
        converter = BaseLegacyConverter(
            legacy_folder_path=Path(__file__).parent,
            target_file_folder=Path(__file__).parent.parent,
            model_reference_category=model_category,
            debug_mode=True,
            print_errors=True,
        )
        converter.normalize_and_convert()
