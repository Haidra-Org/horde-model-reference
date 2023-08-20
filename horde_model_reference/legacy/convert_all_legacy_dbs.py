from pathlib import Path

from horde_model_reference import BASE_PATH, LEGACY_REFERENCE_FOLDER, MODEL_REFERENCE_CATEGORY
from horde_model_reference.legacy.classes.legacy_converters import (
    BaseLegacyConverter,
    LegacyClipConverter,
    LegacyStableDiffusionConverter,
)


def convert_all_legacy_model_references(
    legacy_path: str | Path = LEGACY_REFERENCE_FOLDER,
    target_path: str | Path = BASE_PATH,
) -> bool:
    """
    Convert all legacy model references in the specified folder to the new format.

    Args:
        legacy_path: The path to the folder containing the legacy model references.
        target_path: The path to the folder where the converted model references should be saved.

    Returns:
        True if all conversions succeeded, False otherwise.
    """
    all_succeeded = True

    # Convert stable diffusion model references
    sd_converter = LegacyStableDiffusionConverter(
        legacy_folder_path=legacy_path,
        target_file_folder=target_path,
        debug_mode=False,
    )
    sd_succeeded = sd_converter.normalize_and_convert()
    all_succeeded = all_succeeded and sd_succeeded

    # Convert clip model references
    clip_converter = LegacyClipConverter(
        legacy_folder_path=legacy_path,
        target_file_folder=target_path,
        debug_mode=False,
    )
    clip_succeeded = clip_converter.normalize_and_convert()
    all_succeeded = all_succeeded and clip_succeeded

    # Convert other model references
    non_generic_converter_categories = [
        MODEL_REFERENCE_CATEGORY.stable_diffusion,
        MODEL_REFERENCE_CATEGORY.clip,
    ]
    generic_converted_categories = [x for x in MODEL_REFERENCE_CATEGORY if x not in non_generic_converter_categories]
    for model_category in generic_converted_categories:
        converter = BaseLegacyConverter(
            legacy_folder_path=legacy_path,
            target_file_folder=target_path,
            model_reference_category=model_category,
            debug_mode=False,
        )
        other_succeeded = converter.normalize_and_convert()
        all_succeeded = all_succeeded and other_succeeded

    return all_succeeded


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

    convert_all_legacy_model_references(args.legacy_path, args.target_path)
