import argparse
import json
from pathlib import Path

from horde_model_reference.legacy.classes.raw_legacy_model_database_records import (
    RawLegacy_StableDiffusion_ModelRecord,
)


def validate_legacy_stable_diffusion_db(sd_db: Path, write_to_path: Path | None = None) -> bool:
    raw_json_sd_db: str
    with open(sd_db) as sd_db_file:
        raw_json_sd_db = sd_db_file.read()
    try:
        loaded_json_sd_db = json.loads(raw_json_sd_db)
    except Exception as e:
        print(e)
        print()
        print(f"ERROR: The stable diffusion database specified ({sd_db}) is not a valid json file.")
        if __name__ == "__main__":
            exit(1)
        else:
            return False

    parsed_db_records: dict[str, RawLegacy_StableDiffusion_ModelRecord] = {
        k: RawLegacy_StableDiffusion_ModelRecord.parse_obj(v) for k, v in loaded_json_sd_db.items()
    }

    correct_json_layout = json.dumps({k: v.dict() for k, v in parsed_db_records.items()}, indent=4)
    correct_json_layout += "\n"  # Add a newline to the end of the file, for consistency with formatters.

    if raw_json_sd_db != correct_json_layout:
        print("ERROR: Invalid stable diffusion model database.")
        if write_to_path:
            print(f"Writing the correct stable diffusion model database json to {write_to_path}")
            with open(write_to_path, "w") as corrected_sd_db_file:
                corrected_sd_db_file.write(correct_json_layout)
        else:
            print(
                (
                    "Use the '--write {filename}' command line option to write the corrected "
                    "stable diffusion model database json to a file."
                ),
            )
        if __name__ == "__main__":
            exit(1)
        else:
            return False

    print("Success! Validated stable diffusion model database.")
    if write_to_path:
        print(f"The stable diffusion model database json was already valid, so no file was written to {write_to_path}")

    return True


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.description = "Validate the stable diffusion model database."
    argParser.add_argument(
        "sd_db",
        help="Path to the stable diffusion model database (should be a .json file)",
    )
    argParser.add_argument(
        "--write",
        help="Write the validated database to the specified path, if it fails validation.",
    )
    args = argParser.parse_args()

    validated_sd_db: Path
    try:
        validated_sd_db = Path(args.sd_db)
    except Exception as e:
        print(f"ERROR with --sd_db: {e}")
        print(f"Invalid path: {args.sd_db}")
        exit(1)

    if not validated_sd_db.exists():
        print(f"Path to stable diffusion model database does not exist: {validated_sd_db}")
        exit(1)

    validated_file_output_path: Path | None = None
    if args.write:
        try:
            validated_file_output_path = Path(args.write)
        except Exception as e:
            print(f"ERROR with --write: {e}")
            print(f"Invalid path: {args.write}")
            exit(1)

    validate_legacy_stable_diffusion_db(sd_db=validated_sd_db, write_to_path=validated_file_output_path)
