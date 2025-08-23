import urllib.parse
from urllib.parse import ParseResult

from horde_model_reference import horde_model_reference_paths, meta_consts


def test_github_urls() -> None:
    """Iterates through all github urls in path_consts and asserts they are valid."""
    for url in horde_model_reference_paths.legacy_image_model_github_urls.values():
        parsed_url: ParseResult = urllib.parse.urlparse(url)
        assert parsed_url.scheme in ["http", "https"]
        assert parsed_url.netloc == "raw.githubusercontent.com"


def test_get_all_model_reference_file_paths() -> None:
    """Test the retrieval of all model reference file paths."""
    all_model_reference_file_paths = horde_model_reference_paths.get_all_model_reference_file_paths()
    assert len(all_model_reference_file_paths) > 0

    for model_reference_category in meta_consts.MODEL_REFERENCE_CATEGORY:
        assert model_reference_category.value in all_model_reference_file_paths


def test_stable_diffusion_mapped_to_new_name() -> None:
    """Test that the stable diffusion model reference category is mapped to the new name."""
    from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

    image_generation = str(MODEL_REFERENCE_CATEGORY.image_generation)

    assert image_generation == MODEL_REFERENCE_CATEGORY.image_generation

    from horde_model_reference.model_reference_records import (
        MODEL_RECORD_TYPE_LOOKUP,
        MODEL_REFERENCE_CATEGORY_TYPE_LOOKUP,
    )

    assert (
        MODEL_RECORD_TYPE_LOOKUP[MODEL_REFERENCE_CATEGORY.image_generation]
        == MODEL_RECORD_TYPE_LOOKUP[MODEL_REFERENCE_CATEGORY.image_generation]
    )

    assert (
        MODEL_REFERENCE_CATEGORY_TYPE_LOOKUP[MODEL_REFERENCE_CATEGORY.image_generation]
        == MODEL_REFERENCE_CATEGORY_TYPE_LOOKUP[MODEL_REFERENCE_CATEGORY.image_generation]
    )
