from pathlib import Path

from loguru import logger

from horde_model_reference import MODEL_REFERENCE_CATEGORY, horde_model_reference_paths
from horde_model_reference.legacy.classes.legacy_converters import (
    BaseLegacyConverter,
    LegacyClipConverter,
    LegacyControlnetConverter,
    LegacyStableDiffusionConverter,
    LegacyTextGenerationConverter,
)


def convert_all_legacy_model_references(
    legacy_path: str | Path = horde_model_reference_paths.legacy_path,
    target_path: str | Path = horde_model_reference_paths.base_path,
) -> bool:
    """Convert all legacy model references in the specified folder to the new format.

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

    try:
        sd_converter.convert_to_new_format()
    except Exception as e:
        print(f"Error converting stable diffusion models: {e}")
    all_succeeded = all_succeeded and sd_converter.converted_successfully

    # Convert clip model references
    clip_converter = LegacyClipConverter(
        legacy_folder_path=legacy_path,
        target_file_folder=target_path,
        debug_mode=False,
    )
    try:
        clip_converter.convert_to_new_format()
    except Exception as e:
        print(f"Error converting clip models: {e}")
    all_succeeded = all_succeeded and clip_converter.converted_successfully

    text_converter = LegacyTextGenerationConverter(
        legacy_folder_path=legacy_path,
        target_file_folder=target_path,
        debug_mode=False,
    )
    try:
        text_converter.convert_to_new_format()
    except Exception as e:
        print(f"Error converting text models: {e}")
    all_succeeded = all_succeeded and text_converter.converted_successfully

    # Convert controlnet model references
    controlnet_converter = LegacyControlnetConverter(
        legacy_folder_path=legacy_path,
        target_file_folder=target_path,
        debug_mode=False,
    )
    try:
        controlnet_converter.convert_to_new_format()
    except Exception as e:
        print(f"Error converting controlnet models: {e}")
    all_succeeded = all_succeeded and controlnet_converter.converted_successfully

    # Convert other model references
    non_generic_converter_categories = [
        MODEL_REFERENCE_CATEGORY.image_generation,
        MODEL_REFERENCE_CATEGORY.clip,
        MODEL_REFERENCE_CATEGORY.text_generation,
        MODEL_REFERENCE_CATEGORY.controlnet,
    ]

    generic_converted_categories = [x for x in MODEL_REFERENCE_CATEGORY if x not in non_generic_converter_categories]
    for model_category in generic_converted_categories:
        converter = BaseLegacyConverter(
            legacy_folder_path=legacy_path,
            target_file_folder=target_path,
            model_reference_category=model_category,
            debug_mode=False,
        )
        try:
            converter.convert_to_new_format()
        except Exception as e:
            print(f"Error converting {model_category} models: {e}")
        all_succeeded = all_succeeded and converter.converted_successfully

    logger.info("Finished converting all legacy model references.")
    return all_succeeded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert legacy model reference databases to new format")
    parser.add_argument(
        "--legacy_path",
        type=str,
        default=horde_model_reference_paths.legacy_path,
        help="Path to legacy model reference databases",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default=horde_model_reference_paths.base_path,
        help="Path to save converted model reference databases",
    )
    args = parser.parse_args()

    convert_all_legacy_model_references(args.legacy_path, args.target_path)
