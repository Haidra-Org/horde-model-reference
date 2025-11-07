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
from horde_model_reference.path_consts import normalize_legacy_base_path


def convert_legacy_stable_diffusion_database(
    legacy_path: str | Path = horde_model_reference_paths.legacy_path,
    target_path: str | Path = horde_model_reference_paths.base_path,
) -> bool:
    """Convert the legacy stable diffusion database to the new format.

    Args:
        legacy_path: The path to the folder containing the legacy stable diffusion database.
        target_path: The path to the folder where the converted stable diffusion database should be saved.

    Returns:
        True if the conversion succeeded, False otherwise.
    """
    base_path = normalize_legacy_base_path(legacy_path)
    target_base_path = normalize_legacy_base_path(target_path)

    sd_converter = LegacyStableDiffusionConverter(
        legacy_folder_path=base_path,
        target_file_folder=target_base_path,
        debug_mode=False,
    )
    try:
        sd_converter.convert_to_new_format()
    except Exception as e:
        logger.error(f"Error converting stable diffusion models: {e}")
    return sd_converter.converted_successfully


def convert_legacy_clip_database(
    legacy_path: str | Path = horde_model_reference_paths.legacy_path,
    target_path: str | Path = horde_model_reference_paths.base_path,
) -> bool:
    """Convert the legacy clip database to the new format.

    Args:
        legacy_path: The path to the folder containing the legacy clip database.
        target_path: The path to the folder where the converted clip database should be saved.

    Returns:
        True if the conversion succeeded, False otherwise.
    """
    base_path = normalize_legacy_base_path(legacy_path)
    target_base_path = normalize_legacy_base_path(target_path)

    clip_converter = LegacyClipConverter(
        legacy_folder_path=base_path,
        target_file_folder=target_base_path,
        debug_mode=False,
    )
    try:
        clip_converter.convert_to_new_format()
    except Exception as e:
        logger.error(f"Error converting clip models: {e}")
    return clip_converter.converted_successfully


def convert_legacy_text_generation_database(
    legacy_path: str | Path = horde_model_reference_paths.legacy_path,
    target_path: str | Path = horde_model_reference_paths.base_path,
) -> bool:
    """Convert the legacy text generation database to the new format.

    Args:
        legacy_path: The path to the folder containing the legacy text generation database.
        target_path: The path to the folder where the converted text generation database should be saved.

    Returns:
        True if the conversion succeeded, False otherwise.
    """
    base_path = normalize_legacy_base_path(legacy_path)
    target_base_path = normalize_legacy_base_path(target_path)

    text_converter = LegacyTextGenerationConverter(
        legacy_folder_path=base_path,
        target_file_folder=target_base_path,
        debug_mode=False,
    )
    try:
        text_converter.convert_to_new_format()
    except Exception as e:
        logger.error(f"Error converting text models: {e}")
    return text_converter.converted_successfully


def convert_legacy_controlnet_database(
    legacy_path: str | Path = horde_model_reference_paths.legacy_path,
    target_path: str | Path = horde_model_reference_paths.base_path,
) -> bool:
    """Convert the legacy controlnet database to the new format.

    Args:
        legacy_path: The path to the folder containing the legacy controlnet database.
        target_path: The path to the folder where the converted controlnet database should be saved.

    Returns:
        True if the conversion succeeded, False otherwise.
    """
    base_path = normalize_legacy_base_path(legacy_path)
    target_base_path = normalize_legacy_base_path(target_path)

    controlnet_converter = LegacyControlnetConverter(
        legacy_folder_path=base_path,
        target_file_folder=target_base_path,
        debug_mode=False,
    )
    try:
        controlnet_converter.convert_to_new_format()
    except Exception as e:
        logger.error(f"Error converting controlnet models: {e}")
    return controlnet_converter.converted_successfully


def convert_legacy_database_by_category(
    model_category: MODEL_REFERENCE_CATEGORY,
    legacy_path: str | Path = horde_model_reference_paths.legacy_path,
    target_path: str | Path = horde_model_reference_paths.base_path,
) -> bool:
    """Convert the legacy database for a specific model category to the new format.

    Args:
        model_category: The model reference category to convert.
        legacy_path: The path to the folder containing the legacy database.
        target_path: The path to the folder where the converted database should be saved.

    Returns:
        True if the conversion succeeded, False otherwise.
    """
    normalized_legacy_path = normalize_legacy_base_path(legacy_path)
    normalized_target_path = normalize_legacy_base_path(target_path)

    if model_category == MODEL_REFERENCE_CATEGORY.image_generation:
        return convert_legacy_stable_diffusion_database(normalized_legacy_path, normalized_target_path)

    if model_category == MODEL_REFERENCE_CATEGORY.clip:
        return convert_legacy_clip_database(normalized_legacy_path, normalized_target_path)

    if model_category == MODEL_REFERENCE_CATEGORY.text_generation:
        return convert_legacy_text_generation_database(normalized_legacy_path, normalized_target_path)

    if model_category == MODEL_REFERENCE_CATEGORY.controlnet:
        return convert_legacy_controlnet_database(normalized_legacy_path, normalized_target_path)

    base_converter = BaseLegacyConverter(
        legacy_folder_path=normalized_legacy_path,
        target_file_folder=normalized_target_path,
        model_reference_category=model_category,
        debug_mode=False,
    )
    try:
        base_converter.convert_to_new_format()
    except Exception as e:
        logger.error(f"Error converting {model_category} models: {e}")
    return base_converter.converted_successfully


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

    normalized_legacy_path = normalize_legacy_base_path(legacy_path)
    normalized_target_path = normalize_legacy_base_path(target_path)

    for category in MODEL_REFERENCE_CATEGORY:
        logger.info(f"Converting legacy database for category: {category}")
        succeeded = convert_legacy_database_by_category(category, normalized_legacy_path, normalized_target_path)
        if not succeeded:
            all_succeeded = False
            logger.error(f"Conversion failed for category: {category}")
        else:
            logger.info(f"Successfully converted category: {category}")

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
