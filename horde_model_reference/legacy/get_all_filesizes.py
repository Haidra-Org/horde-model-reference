import json
from pathlib import Path

from loguru import logger

from horde_model_reference.legacy.classes.raw_legacy_model_database_records import (
    RawLegacy_FileRecord,
    RawLegacy_StableDiffusion_ModelRecord,
)
from horde_model_reference.path_consts import AIWORKER_CACHE_HOME


def get_all_file_sizes(sd_db: Path, write_to_path: Path | str) -> bool:

    if AIWORKER_CACHE_HOME is None:
        logger.error("AIWORKER_CACHE_HOME is not set.")
        return False

    raw_json_sd_db: str
    with open(sd_db) as sd_db_file:
        raw_json_sd_db = sd_db_file.read()
    try:
        loaded_json_sd_db = json.loads(raw_json_sd_db)
    except Exception as e:
        logger.exception(e)
        logger.exception(f"ERROR: The stable diffusion database specified ({sd_db}) is not a valid json file.")
        if __name__ == "__main__":
            exit(1)
        else:
            return False

    parsed_db_records: dict[str, RawLegacy_StableDiffusion_ModelRecord] = {
        k: RawLegacy_StableDiffusion_ModelRecord.model_validate(v) for k, v in loaded_json_sd_db.items()
    }

    for _, model_details in parsed_db_records.items():
        if not isinstance(model_details.config["files"][0], RawLegacy_FileRecord):
            logger.error(f"File {model_details.config['files'][0]} is not a valid file record.")
            continue

        filename = model_details.config["files"][0].path

        full_file_path = Path(AIWORKER_CACHE_HOME) / "compvis" / filename
        if not full_file_path.exists():
            logger.error(f"File {full_file_path} does not exist.")
            continue

        model_details.size_on_disk_bytes = full_file_path.stat().st_size

    correct_json_layout = json.dumps(
        {
            k: v.model_dump(
                exclude_none=True,
                exclude_defaults=False,
                by_alias=True,
            )
            for k, v in parsed_db_records.items()
        },
        indent=4,
    )
    correct_json_layout += "\n"  # Add a newline to the end of the file, for consistency with formatters.

    with open(write_to_path, "w") as corrected_sd_db_file:
        corrected_sd_db_file.write(correct_json_layout)
    return True


if __name__ == "__main__":
    get_all_file_sizes(
        Path("t:/_NATAILI_CACHE_HOME_/horde_model_reference/legacy/stable_diffusion.json"),
        "with_sizes.json",
    )
