from pathlib import Path

from horde_model_reference.legacy.classes.legacy_converters import (
    BaseLegacyConverter,
    LegacyClipConverter,
    LegacyStableDiffusionConverter,
)
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORIES

if __name__ == "__main__":
    sd_converter = LegacyStableDiffusionConverter(
        legacy_folder_path=Path(__file__).parent,
        target_file_folder=Path(__file__).parent.parent,
        debug_mode=False,
        print_errors=True,
    )
    sd_converter.normalize_and_convert()

    clip_converter = LegacyClipConverter(
        legacy_folder_path=Path(__file__).parent,
        target_file_folder=Path(__file__).parent.parent,
        debug_mode=False,
        print_errors=True,
    )
    clip_converter.normalize_and_convert()

    non_generic_converter_categories = [
        MODEL_REFERENCE_CATEGORIES.STABLE_DIFFUSION,
        MODEL_REFERENCE_CATEGORIES.CLIP,
    ]

    generic_converted_categories = [x for x in MODEL_REFERENCE_CATEGORIES if x not in non_generic_converter_categories]

    for model_category in generic_converted_categories:
        converter = BaseLegacyConverter(
            legacy_folder_path=Path(__file__).parent,
            target_file_folder=Path(__file__).parent.parent,
            model_reference_category=model_category,
            debug_mode=False,
            print_errors=True,
        )
        converter.normalize_and_convert()
