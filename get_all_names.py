import json

from loguru import logger

from horde_model_reference.model_reference_records import StableDiffusion_ModelRecord

parsed_db_records: dict[str, StableDiffusion_ModelRecord] | None = None
with open("horde_model_reference/stable_diffusion.json") as f:
    sd_model_database_file_json = json.load(f)
    parsed_db_records = {
        k: StableDiffusion_ModelRecord.model_validate(v) for k, v in sd_model_database_file_json.items()
    }


if not parsed_db_records:
    raise RuntimeError("Failed to parse stable diffusion model database.")

logger.debug(len(parsed_db_records))

all_names = [v.name for v in parsed_db_records.values()]
all_names.sort()
logger.debug(json.dumps(all_names))
