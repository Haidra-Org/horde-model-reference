"""Working with records: type narrowing, classification, downloads, serialization.

Run with:

    uv run examples/03_working_with_records.py

`get_model_reference()` is typed as the generic record, but the query API
returns the category-specific record type, which exposes the extra fields.
"""

import json

from horde_model_reference import MODEL_REFERENCE_CATEGORY, ModelReferenceManager


def main() -> None:
    """Inspect a typed record: classification, downloads, and serialization."""
    manager = ModelReferenceManager()

    # The query builder narrows the record type: each item here is an
    # ImageGenerationModelRecord, so `.baseline` / `.nsfw` are statically known.
    image_models = manager.query(MODEL_REFERENCE_CATEGORY.image_generation).limit(1).to_list()
    if not image_models:
        print("No image generation models available.")
        return

    record = image_models[0]
    print(f"Record: {record.name}")
    print(f"  classification: {record.model_classification}")
    print(f"  baseline:       {record.baseline}")
    print(f"  nsfw:           {record.nsfw}")

    # Download metadata is available on the generic base record.
    print(f"  primary URL:    {record.primary_download_url}")
    print(f"  all URLs:       {record.all_download_urls}")

    # Records are Pydantic models, so serialization is built in.
    as_dict = record.model_dump(mode="json", exclude_none=True)
    print("\nSerialized (first 400 chars of JSON):")
    print(json.dumps(as_dict, indent=2)[:400])


if __name__ == "__main__":
    main()
