import glob
import json

from pydantic import ValidationError

from horde_model_reference.legacy.legacy_format import (
    Legacy_ModelDatabaseEntry,
    Legacy_DownloadRecord,
    Legacy_ConfigFileRecord,
    Legacy_StableDiffusionModelReference,
)


from horde_model_reference.model_database_records import (
    StableDiffusionModelReference as New_StableDiffusionModelReference,
)


from pathlib import Path
import urllib.parse
import re

legacyPath = Path("horde_model_reference").joinpath("legacy")
basePath = Path("horde_model_reference")

# githubRepoURL = "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/comfy/"
githubRepoURL = (
    "https://raw.githubusercontent.com/tazlin/AI-Horde-image-model-reference/librarize-modeldb/horde_model_reference/"
)


def model_name_to_showcase_folder_name(showcase_name: str) -> str:
    showcase_name = showcase_name.lower()
    showcase_name = showcase_name.replace("'", "")
    showcase_name = re.sub(r"[^a-z0-9]", "_", showcase_name)
    return showcase_name


def test_convert_legacy_model_database():
    json_data = {}
    with open(legacyPath.joinpath("stable_diffusion.json")) as new_model_reference_file:
        json_data = json.load(new_model_reference_file)

    all_baseline_types = []
    all_styles = []
    all_tags = []
    all_model_hosts = []

    all_records = {}

    existing_showcase_folders = glob.glob(Path("horde_model_reference/showcase/*").__str__(), recursive=True)
    existing_showcase_files = {}
    for showcase_folder in existing_showcase_folders:
        showcase_files = glob.glob(Path(showcase_folder).joinpath("*").__str__(), recursive=True)
        showcase_folder_name = model_name_to_showcase_folder_name(Path(showcase_folder).name)

        existing_showcase_files[showcase_folder_name] = showcase_files

    for model_record_key, model_record_contents in json_data.items():
        new_record = None

        new_record_config_files_list = []
        new_record_config_download_list = []

        try:
            if len(model_record_contents["config"]) > 2:
                print(f"{model_record_key} has more than 2 config entries.")
            sha_lookup = {}
            for config_entry in model_record_contents["config"]:
                if config_entry == "files":
                    for config_file in model_record_contents["config"][config_entry]:
                        parsed_file_record = Legacy_ConfigFileRecord(**config_file)
                        if ".yaml" in parsed_file_record.path:
                            continue
                        sha_lookup[parsed_file_record.path] = parsed_file_record.sha256sum
                        parsed_file_record.sha256sum = None
                        new_record_config_files_list.append(parsed_file_record)
                elif config_entry == "download":
                    for download in model_record_contents["config"][config_entry]:
                        parsed_download_record = Legacy_DownloadRecord(**download)
                        parsed_download_record.sha256sum = sha_lookup[parsed_download_record.file_name]
                        new_record_config_download_list.append(parsed_download_record)

            model_record_contents["config"] = {
                # "files": new_record_config_files_list,
                "download": new_record_config_download_list,
            }
            new_record = Legacy_ModelDatabaseEntry(**model_record_contents)
            all_records[model_record_key] = new_record
        except ValidationError as error:
            print("*" * 80)
            print(f"Error parsing {model_record_key}:\n{error}")
            print(f"{model_record_contents=}")
            print("skipping.")
            print("*" * 80)
            continue

        if new_record is None:
            continue

        if new_record.name != model_record_key:
            print(f"name mismatch for {model_record_key}.")

        if new_record.baseline == "stable diffusion 1":
            new_record.baseline = "stable_diffusion_1"
            # new_record.baseline_trained_resolution = 256
        elif new_record.baseline == "stable diffusion 2":
            new_record.baseline = "stable_diffusion_2_768"
        elif new_record.baseline == "stable diffusion 2 512":
            new_record.baseline = "stable_diffusion_2_512"

        if new_record.baseline not in all_baseline_types:
            all_baseline_types.append(new_record.baseline)
            print(f"-> Found new baseline type: {new_record.baseline}.")

        if new_record.available:
            print(f"{model_record_key} is available.")

        if new_record.download_all:
            # print(f"{model_record_key} has download_all set.")
            pass

        if new_record.config is None:
            print(f"{model_record_key} has no config.")

        if new_record.showcases is None or len(new_record.showcases) == 0:
            pass
            # print(f"{model_record_key} has no showcases.")
        else:
            if any("huggingface" in showcase for showcase in new_record.showcases):
                print(f"{model_record_key} has a huggingface showcase.")

            expected_showcase_foldername = model_name_to_showcase_folder_name(model_record_key)
            if expected_showcase_foldername not in existing_showcase_files:
                print(f"{model_record_key} has no showcase folder.")
                print(f"{expected_showcase_foldername=}")
            new_record.showcases = []
            for file in existing_showcase_files[expected_showcase_foldername]:
                url_friendly_name = urllib.parse.quote(Path(file).name)
                # if not any(url_friendly_name in showcase for showcase in new_record.showcases):
                #     print(f"{model_record_key} is missing a showcase for {url_friendly_name}.")
                #     print(f"{new_record.showcases=}")
                #     continue
                expected_github_location = urllib.parse.urljoin(
                    githubRepoURL, f"showcase/{expected_showcase_foldername}/{url_friendly_name}"
                )
                new_record.showcases.append(expected_github_location)

        if new_record.description is None:
            print(f"{model_record_key} has no description.")

        if new_record.style == "":
            print(f"{model_record_key} has no style.")
        elif new_record.style not in all_styles:
            all_styles.append(new_record.style)
            # print(f"-> Found new style: {new_record.style}.")

        if new_record.type != "ckpt":
            print(f"{model_record_key} is not a ckpt.")

        if new_record.tags is not None:
            for tag in new_record.tags:
                if tag not in all_tags:
                    all_tags.append(tag)
                    # print(f"-> Found new tag: {tag}.")

        if len(new_record.config) == 0:
            print(f"{model_record_key} has no config.")

        for config_entry in new_record.config:
            if config_entry == "files":
                for config_file in new_record.config[config_entry]:
                    if config_file.path is None or config_file.path == "":
                        print(f"{model_record_key} has a config file with no path.")

                    if ".yaml" in config_file.path:
                        if config_file.path != "v2-inference-v.yaml" and config_file.path != "v1-inference.yaml":
                            print(f"{model_record_key} has a non-standard config.")
                        continue
                    elif ".ckpt" not in config_file.path:
                        print(f"{model_record_key} does not have a ckpt file specified.")

                    # if config_file.md5sum is None or config_file.md5sum == "":
                    #     print(f"{model_record_key} has a config file with no md5sum.")
                    # else:
                    #     if len(config_file.md5sum) != 32:
                    #         print(f"{model_record_key} has a config file with an invalid md5sum.")

                    if config_file.sha256sum is None or config_file.sha256sum == "":
                        print(f"{model_record_key} has a config file with no sha256sum.")
                    else:
                        if len(config_file.sha256sum) != 64:
                            print(f"{model_record_key} has a config file with an invalid sha256sum.")

            elif config_entry == "download":
                for download in new_record.config[config_entry]:
                    if download.file_name is None or download.file_name == "":
                        print(f"{model_record_key} has a download with no file_name.")

                    if download.file_path is None or download.file_path != "":
                        print(f"{model_record_key} has a download with a file_path.")

                    if download.file_url is None or download.file_url == "":
                        print(f"{model_record_key} has a download with no file_url.")
                        continue

                    if "civitai" in download.file_url:
                        # print(
                        #     f"{model_record_key} has a download with a civitai file_url.",
                        # )
                        download.known_slow_download = True

                    parsedURL = urllib.parse.urlparse(download.file_url)

                    if parsedURL.hostname not in all_model_hosts:
                        all_model_hosts.append(parsedURL.hostname)

    print(f"{all_styles=}")
    print(f"{all_baseline_types=}")
    print(f"{all_tags=}")

    print(f"{len(all_records)=}")

    modelReference = Legacy_StableDiffusionModelReference(
        baseline_types=all_baseline_types,
        styles=all_styles,
        tags=all_tags,
        model_hosts=all_model_hosts,
        models=all_records,
    )

    with open(
        Path("test.json"),
        "w",
    ) as testfile:
        testfile.write(modelReference.json(indent=4, exclude_defaults=True, exclude_none=True, exclude_unset=True))


def test_validate_converted_model_database():
    model_reference = New_StableDiffusionModelReference.parse_file(Path("test.json"))

    assert len(model_reference.baseline_types) >= 3
    for baseline_type in model_reference.baseline_types:
        assert baseline_type != ""

    assert len(model_reference.styles) >= 6
    for style in model_reference.styles:
        assert style != ""

    assert len(model_reference.model_hosts) >= 1
    for model_host in model_reference.model_hosts:
        assert model_host != ""

    assert model_reference.models is not None
    assert len(model_reference.models) >= 100

    assert model_reference.models["stable_diffusion"] is not None
    assert model_reference.models["stable_diffusion"].name == "stable_diffusion"
    assert model_reference.models["stable_diffusion"].showcases is not None
    assert len(model_reference.models["stable_diffusion"].showcases) >= 3

    for model_key, model_info in model_reference.models.items():

        assert model_info.name == model_key
        assert model_info.baseline in model_reference.baseline_types
        assert model_info.style in model_reference.styles

        if model_info.homepage is not None:
            assert model_info.homepage != ""
            parsedHomepage = urllib.parse.urlparse(model_info.homepage)
            assert parsedHomepage.scheme == "https" or parsedHomepage.scheme == "http"

        assert model_info.description is not None
        assert model_info.description != ""
        assert model_info.version != ""

        if model_info.tags is not None:
            for tag in model_info.tags:
                assert tag in model_reference.tags

        for config_key, config_section in model_info.config.items():
            assert config_key != "files"

            if config_key == "download":
                for download_record in config_section:
                    assert download_record.file_name is not None
                    assert download_record.file_url is not None
            else:
                assert False

        if model_info.trigger is not None:
            for trigger_record in model_info.trigger:
                assert trigger_record != ""
