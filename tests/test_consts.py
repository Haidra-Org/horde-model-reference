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


def test_classifications() -> None:
    """Test that all model reference categories have valid classifications."""
    model_classifications = meta_consts.MODEL_CLASSIFICATION_LOOKUP

    for category, classification in model_classifications.items():
        if "generation" in category:
            assert classification.purpose == meta_consts.MODEL_PURPOSE.generation


def test_get_model_name_variants_includes_sanitized_koboldcpp_variant() -> None:
    """Ensure canonical names with org prefixes map to flattened koboldcpp variants."""
    canonical_name = "Qwen/Qwen3-4B-Instruct-2507"

    variants = meta_consts.get_model_name_variants(canonical_name)

    assert "koboldcpp/Qwen3-4B-Instruct-2507" in variants
    assert "koboldcpp/Qwen_Qwen3-4B-Instruct-2507" in variants


def test_get_model_name_variants_does_not_duplicate_without_org() -> None:
    """Ensure flattened variants are not duplicated when no org prefix exists."""
    variants = meta_consts.get_model_name_variants("Broken-Tutu-24B")

    assert variants.count("koboldcpp/Broken-Tutu-24B") == 1
