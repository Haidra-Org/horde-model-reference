import urllib.parse

from horde_model_reference import meta_consts, path_consts


def test_github_urls():
    for url in path_consts.LEGACY_MODEL_GITHUB_URLS.values():
        parsed_url = urllib.parse.urlparse(url)
        print(parsed_url.geturl())
        assert parsed_url.scheme in ["http", "https"]
        assert parsed_url.netloc == "raw.githubusercontent.com"
