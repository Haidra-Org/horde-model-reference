import urllib.parse
from urllib.parse import ParseResult

from horde_model_reference import meta_consts, path_consts


def test_github_urls() -> None:
    """Iterates through all github urls in path_consts and asserts they are valid."""
    for url in path_consts.LEGACY_MODEL_GITHUB_URLS.values():
        parsed_url: ParseResult = urllib.parse.urlparse(url)
        assert parsed_url.scheme in ["http", "https"]
        assert parsed_url.netloc == "raw.githubusercontent.com"


def test_get_all_model_reference_file_paths() -> None:
    all_model_reference_file_paths = path_consts.get_all_model_reference_file_paths()
    assert len(all_model_reference_file_paths) > 0

    for model_reference_category in meta_consts.MODEL_REFERENCE_CATEGORY:
        assert model_reference_category.value in all_model_reference_file_paths


def test_stable_diffusion_mapped_to_new_name() -> None:

    from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY

    image_generation = str(MODEL_REFERENCE_CATEGORY.image_generation)

    assert image_generation == MODEL_REFERENCE_CATEGORY.stable_diffusion

    from horde_model_reference.model_reference_records import (
        MODEL_REFERENCE_RECORD_TYPE_LOOKUP,
        MODEL_REFERENCE_CATEGORY_TYPE_LOOKUP,
    )

    assert (
        MODEL_REFERENCE_RECORD_TYPE_LOOKUP[MODEL_REFERENCE_CATEGORY.image_generation]
        == MODEL_REFERENCE_RECORD_TYPE_LOOKUP[MODEL_REFERENCE_CATEGORY.stable_diffusion]
    )

    assert (
        MODEL_REFERENCE_CATEGORY_TYPE_LOOKUP[MODEL_REFERENCE_CATEGORY.image_generation]
        == MODEL_REFERENCE_CATEGORY_TYPE_LOOKUP[MODEL_REFERENCE_CATEGORY.stable_diffusion]
    )
