"""Script to add models to the legacy stable diffusion model reference."""

import argparse
import glob
import hashlib
import json
import os
import pathlib
import urllib.parse

from loguru import logger

from horde_model_reference import (
    KNOWN_TAGS,
    LEGACY_REFERENCE_FOLDER,
    MODEL_REFERENCE_CATEGORY,
    MODEL_STYLE,
    get_model_reference_file_path,
)
from horde_model_reference.legacy.classes.raw_legacy_model_database_records import (
    RawLegacy_DownloadRecord,
    RawLegacy_FileRecord,
    RawLegacy_StableDiffusion_ModelRecord,
)


class Legacy_StableDiffusionRecordHelper:
    """A helper class to add models to the legacy stable diffusion model reference."""

    target_folder: pathlib.Path

    def __init__(self, target_folder: str | pathlib.Path):
        if not isinstance(target_folder, pathlib.Path):
            target_folder = pathlib.Path(target_folder)
        self.target_folder = target_folder

    def load_legacy_sd_model_reference(self) -> dict[str, RawLegacy_StableDiffusion_ModelRecord]:
        path_to_legacy_stablediffusion = get_model_reference_file_path(
            MODEL_REFERENCE_CATEGORY.stable_diffusion,
            base_path=LEGACY_REFERENCE_FOLDER,
        )
        sd_legacy_model_reference: dict[str, RawLegacy_StableDiffusion_ModelRecord] = {}
        with open(str(path_to_legacy_stablediffusion)) as f:
            raw_json = json.load(f)

            for raw_record_key, raw_record_contents in raw_json.items():
                try:
                    sd_legacy_model_reference[raw_record_key] = RawLegacy_StableDiffusion_ModelRecord.model_validate(
                        raw_record_contents,
                    )
                except Exception as e:
                    logger.exception(f"Failed to parse {raw_record_key} due to {e}")

        return sd_legacy_model_reference

    def get_sd_models_on_disk(self):
        all_sd_models = []
        all_sd_models.extend(glob.glob(str(f"{self.target_folder}/*.ckpt"), recursive=True))
        all_sd_models.extend(glob.glob(str(f"{self.target_folder}/*.safetensors"), recursive=True))
        return all_sd_models

    def is_in_sd_legacy_model_reference(
        self,
        model_filename: str,
        sd_legacy_model_reference: dict[str, RawLegacy_StableDiffusion_ModelRecord],
    ):
        for legacy_model_record in sd_legacy_model_reference.values():
            download = legacy_model_record.config["download"][0]
            if not isinstance(download, RawLegacy_DownloadRecord):
                raise TypeError(f"Expected download to be a DownloadRecord, got {download}")
            if download.file_name == str(model_filename):
                # print(f"Found {model_on_disk} in legacy model reference")
                return True
        return False

    def add_models_from_disk(self, file_out: pathlib.Path) -> None:
        all_sd_models_on_disk = self.get_sd_models_on_disk()
        sd_legacy_model_reference = self.load_legacy_sd_model_reference()
        new_models: dict[str, RawLegacy_StableDiffusion_ModelRecord] = {}

        for model_on_disk in all_sd_models_on_disk:
            model_on_disk_path = pathlib.Path(model_on_disk)
            if not self.is_in_sd_legacy_model_reference(model_on_disk_path.name, sd_legacy_model_reference):
                print(f"Found {model_on_disk} not in legacy model reference")
                print("Would you like to add it now? (y/n)")
                user_input = input().lower()
                if user_input != "y":
                    print(f"Skipping {model_on_disk}")
                    continue
                print(f"Adding {model_on_disk} to legacy model reference")
                print("Calculating sha256sum of file on disk. This may take several seconds...")
                sha256sum = ""
                with open(model_on_disk, "rb") as model_file:
                    data = model_file.read()
                    sha256sum = hashlib.sha256(data).hexdigest()
                    print(f"sha256sum: {sha256sum}")
                    sha256sum_filename = model_on_disk_path.name.split(".")[0]
                    sha256sum_filename_path = model_on_disk_path.parent.joinpath(f"{sha256sum_filename}.sha256")

                    if sha256sum_filename_path.exists():
                        print()
                        print(f"WARNING: sha256sum file already exists at {sha256sum_filename_path}")
                        print("This probably shouldn't be the case. Please check the file if this unexpected.")
                        print("Would you like to overwrite it? (y/n)")
                        user_input = input().lower()
                        if user_input != "y":
                            print(f"Skipping {model_on_disk}")
                            continue

                    with open(sha256sum_filename_path, "w") as sha256sum_file:
                        sha256sum_file.write(f"{sha256sum} *{sha256sum_filename}")

                print("Please enter the user friendly model name: (e.g. 'Stable Diffusion 1')")
                model_friendly_name = input()

                print(
                    (
                        "Please enter the model description: "
                        "(e.g. 'Generalist AI image generating model. The baseline for all finetuned models.')"
                    ),
                )
                model_description = input()

                print("Please enter the version of the model: (e.g. '1.5')")
                version = input()

                download_url = ""
                while True:
                    print("Please enter the *download* url: (e.g. 'https://example.com/model.ckpt')")
                    download_url = input()
                    parsed_url = None
                    try:
                        parsed_url = urllib.parse.urlparse(download_url)
                        if not parsed_url:
                            print(f"Invalid url: {download_url}")
                            continue

                        if "http" not in parsed_url.scheme:
                            print("Invalid url: scheme must be http or https")
                            continue
                        if not parsed_url.netloc:
                            print("Invalid url: no netloc (domain) specified")
                            continue
                        break
                    except Exception as e:
                        print(f"Invalid url: {download_url}\n{e}")

                homepage_url = ""
                while True:
                    print("Please enter the *homepage* url (NOT the download): (e.g. 'https://example.com/')")
                    homepage_url = input()
                    parsed_url = None
                    try:
                        parsed_url = urllib.parse.urlparse(homepage_url)
                        if not parsed_url:
                            print(f"Invalid url: {homepage_url}")
                            continue

                        if "http" not in parsed_url.scheme:
                            print("Invalid url: scheme must be http or https")
                            continue
                        if not parsed_url.netloc:
                            print("Invalid url: no netloc (domain) specified")
                            continue
                        break
                    except Exception as e:
                        print(f"Invalid url: {homepage_url}\n{e}")

                # ask for baseline type, default to stable diffusion 1
                base_num_lookup = {
                    x + 1: y
                    for x, y in enumerate(["stable diffusion 1", "stable diffusion 2", "stable diffusion 2 512"])
                }
                baseline_type = "stable_diffusion_1"
                while True:
                    print("Please select the baseline type:")
                    for base_num, base_choice_num in base_num_lookup.items():
                        print(f"{base_num}: {base_choice_num}")
                    base_chosen = input()
                    if not base_chosen.isdigit():
                        print("Select default baseline 'stable_diffusion_1' (y/n)")
                        if input().lower() == "y":
                            break
                        continue
                    if int(base_chosen) not in base_num_lookup:
                        print("Invalid baseline")
                        continue
                    break

                print("Is this model nsfw? (y/n)")
                is_nsfw = input().lower() == "y"

                style_num_lookup = {x + 1: y for x, y in enumerate(MODEL_STYLE)}
                style = "generalist"
                while True:
                    print("Please enter the style of the model:")
                    for style_num, style_choice_num in style_num_lookup.items():
                        print(f"{style_num}: {style_choice_num}")
                    style_chosen = input()
                    if not style_chosen.isdigit():
                        print("Select default style 'generalist (y/n)'")
                        if input().lower() == "y":
                            break

                        continue
                    if int(style_chosen) not in style_num_lookup:
                        print("Invalid style")
                        continue
                    break

                triggers: list[str] = []
                while True:
                    print("Enter any triggers, one at a time. Leave the line blank and press enter when finished.")
                    trigger = input()
                    if not trigger:
                        print(f"Trigger(s) ({len(triggers)}): {triggers}")
                        break
                    triggers.append(trigger)

                tags_chosen: list[str] = []
                while True:
                    print("Select any tags, one at a time. Leave the line blank and press enter when finished.")
                    tag_num_lookup = {x + 1: y for x, y in enumerate(KNOWN_TAGS)}
                    for tag_num, tag_choice_num in tag_num_lookup.items():
                        print(f"{tag_num}: {tag_choice_num}")
                    tag_chosen = input()
                    if not tag_chosen.isdigit():
                        print(f"Tag(s) ({len(tags_chosen)}): {len(tags_chosen)}")
                        break
                    if int(tag_chosen) not in tag_num_lookup:
                        print("Invalid tag")
                        continue
                    tags_chosen.append(tag_num_lookup[int(tag_chosen)])

                print("Is this an inpainting model? (y/n)")
                inpainting = input().lower() == "y"

                # FIXME ADD TRIGGERS
                new_file_record = RawLegacy_FileRecord(path=model_on_disk_path.name, sha256sum=sha256sum)
                legacy_yaml_filename = "v1-inference.yaml"
                if baseline_type != "stable_diffusion_1":
                    legacy_yaml_filename = "v2-inference.yaml"

                legacy_yaml_file_record = RawLegacy_FileRecord(path=legacy_yaml_filename)
                new_download_record = RawLegacy_DownloadRecord(
                    file_name=model_on_disk_path.name,
                    file_path="",
                    file_url=download_url,
                )
                new_model_record = RawLegacy_StableDiffusion_ModelRecord(
                    name=model_friendly_name,
                    baseline=baseline_type,
                    type="ckpt",
                    inpainting=inpainting,
                    description=model_description,
                    tags=tags_chosen if len(tags_chosen) > 0 else None,
                    showcases=None,
                    min_bridge_version=None,
                    version=version,
                    style=style,
                    trigger=triggers if len(triggers) > 0 else None,
                    homepage=homepage_url,
                    nsfw=is_nsfw,
                    download_all=False,
                    config={"files": [new_file_record, legacy_yaml_file_record], "download": [new_download_record]},
                    available=False,
                )
                new_models[model_friendly_name] = new_model_record

        if len(new_models) == 0:
            print("No new models found")
        sd_legacy_model_reference.update(new_models)

        jsonable_out = {}
        for model_name, model_record in sd_legacy_model_reference.items():
            jsonable_out[model_name] = model_record.model_dump(
                exclude_none=True,
                exclude_unset=True,
                by_alias=True,
            )

        with open(str(file_out), "w") as f:
            f.write(json.dumps(jsonable_out, indent=4))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--file_out", help="File to write the new models to")
    arg_parser.add_argument(
        "--model_folder",
        action="store_true",
        help="Base folder to look for models in. Should usually be the same as AIWORKER_CACHE_HOME",
    )
    args = arg_parser.parse_args()

    _CACHE_HOME_STR: str = os.environ.get("AIWORKER_CACHE_HOME", "")

    if args.model_folder:
        _CACHE_HOME_STR = args.model_folder

    if not _CACHE_HOME_STR:
        print("AIWORKER_CACHE_HOME not set. Either set it or pass --model_folder")
        exit(1)

    CACHE_HOME_PATH = pathlib.Path(_CACHE_HOME_STR, "nataili")

    if not CACHE_HOME_PATH.exists():
        print(f"Model folder {_CACHE_HOME_STR} does not exist!")
        exit(1)

    STABLE_DIFFUSION_FOLDER = pathlib.Path(CACHE_HOME_PATH, "compvis")

    if not STABLE_DIFFUSION_FOLDER.exists():
        print(f"compvis folder {STABLE_DIFFUSION_FOLDER} does not exist")
        exit(1)

    Legacy_StableDiffusionRecordHelper(STABLE_DIFFUSION_FOLDER).add_models_from_disk(pathlib.Path("test.json"))
