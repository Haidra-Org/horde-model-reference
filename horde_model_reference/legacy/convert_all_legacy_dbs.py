from pathlib import Path

from horde_model_reference import BASE_PATH, LEGACY_REFERENCE_FOLDER, MODEL_REFERENCE_CATEGORIES
from horde_model_reference.legacy.classes.legacy_converters import (
    BaseLegacyConverter,
    LegacyClipConverter,
    LegacyStableDiffusionConverter,
)


def main(legacy_path: str | Path = LEGACY_REFERENCE_FOLDER, target_path: str | Path = BASE_PATH):
    sd_converter = LegacyStableDiffusionConverter(
        legacy_folder_path=legacy_path,
        target_file_folder=target_path,
        debug_mode=False,
    )
    sd_converter.normalize_and_convert()

    clip_converter = LegacyClipConverter(
        legacy_folder_path=legacy_path,
        target_file_folder=target_path,
        debug_mode=False,
    )
    clip_converter.normalize_and_convert()

    non_generic_converter_categories = [
        MODEL_REFERENCE_CATEGORIES.stable_diffusion,
        MODEL_REFERENCE_CATEGORIES.clip,
    ]

    generic_converted_categories = [x for x in MODEL_REFERENCE_CATEGORIES if x not in non_generic_converter_categories]

    for model_category in generic_converted_categories:
        converter = BaseLegacyConverter(
            legacy_folder_path=legacy_path,
            target_file_folder=target_path,
            model_reference_category=model_category,
            debug_mode=False,
        )
        converter.normalize_and_convert()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert legacy model reference databases to new format")
    parser.add_argument(
        "--legacy_path",
        type=str,
        default=LEGACY_REFERENCE_FOLDER,
        help="Path to legacy model reference databases",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default=BASE_PATH,
        help="Path to save converted model reference databases",
    )
    args = parser.parse_args()

    main(args.legacy_path, args.target_path)
