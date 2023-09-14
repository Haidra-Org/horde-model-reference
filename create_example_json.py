import json

from horde_model_reference import MODEL_PURPOSE, MODEL_STYLE, STABLE_DIFFUSION_BASELINE_CATEGORY
from horde_model_reference.legacy.classes.raw_legacy_model_database_records import (
    RawLegacy_StableDiffusion_ModelReference,
)
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    StableDiffusion_ModelRecord,
    StableDiffusion_ModelReference,
)

STABLE_DIFFUSION_EXAMPLE_JSON_FILENAME = "stable_diffusion.example.json"
STABLE_DIFFUSION_SCHEMA_JSON_FILENAME = "stable_diffusion.schema.json"

LEGACY_STABLE_DIFFUSION_SCHEMA_JSON_FILENAME = "legacy_stable_diffusion.schema.json"


def main():
    example_record_name = "Example General Model"
    example_download_record = DownloadRecord(
        file_name="example_general_model.ckpt",
        file_url="https://www.some_website.com/a_different_name_on_the_website.ckpt",
        sha256sum="DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF",
    )

    # An example StableDiffusion_ModelRecord with test data
    example_model_record = StableDiffusion_ModelRecord(
        name=example_record_name,
        description="This would be a description of the model.",
        version="1.0",
        inpainting=False,
        style=MODEL_STYLE.generalist,
        config={"download": [example_download_record]},
        purpose=MODEL_PURPOSE.image_generation,
        baseline=STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_1,
        tags=["anime", "faces"],
        showcases=[
            "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/showcase/test/test_general_01.png",
        ],
        min_bridge_version=12,
        trigger=["trigger1", "some other_trigger"],
        homepage="https://www.not.a.real_website.com",
        nsfw=False,
        size_on_disk_bytes=123456789,
    )

    example_record_2_name = "Example anime model"
    example_download_record_2 = DownloadRecord(
        file_name="example_general_model.ckpt",
        file_url="https://www.some_website.com/a_different_name_on_the_website.ckpt",
        sha256sum="DEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEFDEADBEEF",
    )
    example_model_record_2 = StableDiffusion_ModelRecord(
        name=example_record_2_name,
        description="This would be a description of the model.",
        version="2.5",
        inpainting=False,
        style=MODEL_STYLE.anime,
        config={"download": [example_download_record_2]},
        purpose=MODEL_PURPOSE.image_generation,
        baseline=STABLE_DIFFUSION_BASELINE_CATEGORY.stable_diffusion_1,
        tags=["anime", "faces"],
        showcases=[
            "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/showcase/test/anime_01.png",
        ],
        min_bridge_version=12,
        trigger=["anime", "some other_anime_trigger"],
        homepage="https://www.another_fake_website.com",
        nsfw=True,
        size_on_disk_bytes=123456789,
    )

    reference = StableDiffusion_ModelReference(
        root={
            "example model 1": example_model_record,
            "example model 2": example_model_record_2,
        },
    )

    with open(STABLE_DIFFUSION_EXAMPLE_JSON_FILENAME, "w") as example_file:
        example_file.write(reference.model_dump_json(indent=4) + "\n")

    with open(STABLE_DIFFUSION_SCHEMA_JSON_FILENAME, "w") as schema_file:
        schema_file.write(json.dumps(StableDiffusion_ModelReference.model_json_schema(), indent=4) + "\n")

    with open(LEGACY_STABLE_DIFFUSION_SCHEMA_JSON_FILENAME, "w") as schema_file:
        schema_file.write(json.dumps(RawLegacy_StableDiffusion_ModelReference.model_json_schema(), indent=4) + "\n")


if __name__ == "__main__":
    main()
