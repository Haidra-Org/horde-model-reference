import json

from horde_model_reference.meta_consts import MODEL_PURPOSE, MODEL_STYLES, STABLE_DIFFUSION_BASELINE_CATEGORIES
from horde_model_reference.model_reference_records import (
    DownloadRecord,
    StableDiffusion_ModelRecord,
)

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
    style=MODEL_STYLES.generalist,
    config={"download": [example_download_record]},
    model_purpose=MODEL_PURPOSE.image_generation,
    baseline=STABLE_DIFFUSION_BASELINE_CATEGORIES.stable_diffusion_1,
    tags=["anime", "faces"],
    showcases=[
        "https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/showcase/test/test_general_01.png",
    ],
    min_bridge_version=12,
    trigger=["trigger1", "some other_trigger"],
    homepage="https://www.not.a.real_website.com",
    nsfw=False,
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
    style=MODEL_STYLES.anime,
    config={"download": [example_download_record_2]},
    model_purpose=MODEL_PURPOSE.image_generation,
    baseline=STABLE_DIFFUSION_BASELINE_CATEGORIES.stable_diffusion_1,
    tags=["anime", "faces"],
    showcases=["https://raw.githubusercontent.com/db0/AI-Horde-image-model-reference/main/showcase/test/anime_01.png"],
    min_bridge_version=12,
    trigger=["anime", "some other_anime_trigger"],
    homepage="https://www.another_fake_website.com",
    nsfw=True,
)

jsonable_dict = {
    example_record_name: example_model_record.dict(),
    example_record_2_name: example_model_record_2.dict(),
}
with open("stable_diffusion.example.json", "w") as example_file:
    example_file.write(json.dumps(jsonable_dict, indent=4))

with open("stable_diffusion.schema.json", "w") as schema_file:
    schema_file.write(StableDiffusion_ModelRecord.schema_json(indent=4))
