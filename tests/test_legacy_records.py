import json

from horde_model_reference import MODEL_REFERENCE_CATEGORY, ModelReferenceManager, ReplicateMode
from horde_model_reference.backends import GitHubBackend
from horde_model_reference.legacy.classes.legacy_models import (
    LegacyClipRecord,
    LegacyGenericRecord,
    LegacyStableDiffusionRecord,
)


def test_legacy_record_read_and_init(
    restore_manager_singleton: None,
) -> None:
    """Tests reading and initializing legacy records."""
    model_reference_manager = ModelReferenceManager(replicate_mode=ReplicateMode.REPLICA, lazy_mode=True)

    legacy_reference_locations = model_reference_manager.backend.get_all_category_file_paths()

    assert len(legacy_reference_locations) > 0
    assert MODEL_REFERENCE_CATEGORY.image_generation in legacy_reference_locations

    assert all(ref_cat in legacy_reference_locations for ref_cat in MODEL_REFERENCE_CATEGORY)

    model_reference_manager.backend.fetch_all_categories(force_refresh=True)

    assert isinstance(model_reference_manager.backend, GitHubBackend)

    legacy_references = {}
    for category, file_path in model_reference_manager.backend._references_paths_cache.items():
        if file_path and file_path.exists():
            with open(file_path) as f:
                legacy_references[category] = json.load(f)
        else:
            legacy_references[category] = None

    assert len(legacy_references) > 0
    assert all(ref_cat in legacy_references for ref_cat in MODEL_REFERENCE_CATEGORY)

    legacy_image_model_reference_dict = legacy_references[MODEL_REFERENCE_CATEGORY.image_generation]
    assert legacy_image_model_reference_dict is not None
    assert len(legacy_image_model_reference_dict) > 0

    for image_record_name, image_record in legacy_image_model_reference_dict.items():
        image_parsed_record = LegacyStableDiffusionRecord.model_validate(image_record)
        assert image_parsed_record.name == image_record_name
        assert isinstance(image_parsed_record, LegacyStableDiffusionRecord)
        assert len(image_parsed_record.config.download) > 0
        assert len(image_parsed_record.config.files) > 0

    legacy_clip_model_reference_dict = legacy_references[MODEL_REFERENCE_CATEGORY.clip]
    assert legacy_clip_model_reference_dict is not None
    assert len(legacy_clip_model_reference_dict) > 0
    for clip_record_name, clip_record in legacy_clip_model_reference_dict.items():
        clip_parsed_record = LegacyClipRecord.model_validate(clip_record)
        assert clip_parsed_record.name == clip_record_name
        assert isinstance(clip_parsed_record, LegacyClipRecord)

    legacy_generic_model_reference_dict = legacy_references[MODEL_REFERENCE_CATEGORY.blip]
    assert legacy_generic_model_reference_dict is not None
    assert len(legacy_generic_model_reference_dict) > 0
    for generic_record_name, generic_record in legacy_generic_model_reference_dict.items():
        generic_parsed_record = LegacyGenericRecord.model_validate(generic_record)
        assert generic_parsed_record.name == generic_record_name
        assert isinstance(generic_parsed_record, LegacyGenericRecord)
