import pathlib
import warnings

from horde_model_reference.legacy.legacy_download_manager import LegacyReferenceDownloadManager
from horde_model_reference.meta_consts import MODEL_REFERENCE_CATEGORY
from horde_model_reference.path_consts import HORDE_PROXY_URL_BASE


def download_all_models(
    override_existing: bool = False,
    proxy_url: str = HORDE_PROXY_URL_BASE,
) -> dict[MODEL_REFERENCE_CATEGORY, pathlib.Path | None]:
    reference_dm = LegacyReferenceDownloadManager(proxy_url=proxy_url)
    warnings.warn(
        "download_all_models() is deprecated, use the class `LegacyReferenceDownloadManager` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return reference_dm.download_all_legacy_model_references(overwrite_existing=override_existing)


def main():
    reference_download_manager = LegacyReferenceDownloadManager()
    references_and_paths = reference_download_manager.download_all_legacy_model_references(overwrite_existing=True)

    for reference, path in references_and_paths.items():
        print(f"Downloaded {reference.name}: {path}")


if __name__ == "__main__":
    main()
