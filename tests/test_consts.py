import urllib.parse

from horde_model_reference import path_consts


def test_github_urls():
    """Iterates through all github urls in path_consts and asserts they are valid."""
    for url in path_consts.LEGACY_MODEL_GITHUB_URLS.values():
        parsed_url = urllib.parse.urlparse(url)
        assert parsed_url.scheme in ["http", "https"]
        assert parsed_url.netloc == "raw.githubusercontent.com"
