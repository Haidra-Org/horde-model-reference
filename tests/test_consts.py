import urllib.parse

from horde_model_reference import meta_consts, path_consts


def test_github_urls():
    """Iterates through all github urls in path_consts and asserts they are valid."""
    for url in path_consts.LEGACY_MODEL_GITHUB_URLS.values():
        parsed_url = urllib.parse.urlparse(url)
        assert parsed_url.scheme in ["http", "https"]
        assert parsed_url.netloc == "raw.githubusercontent.com"


def test_get_all_model_reference_file_paths():
    all_model_reference_file_paths = path_consts.get_all_model_reference_file_paths()
    assert len(all_model_reference_file_paths) > 0

    for model_reference_category in meta_consts.MODEL_REFERENCE_CATEGORY:
        assert model_reference_category.value in all_model_reference_file_paths
