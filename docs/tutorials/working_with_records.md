# Working with Model Records

Model records are Pydantic models that represent individual entries in the model reference. This tutorial covers the record hierarchy, key fields per category, and how to work with records in your code.

## Record Hierarchy

All records inherit from `GenericModelRecord`:

```
GenericModelRecord
  +-- ImageGenerationModelRecord
  +-- TextGenerationModelRecord
  +-- ControlNetModelRecord
  +-- ClipModelRecord
  +-- BlipModelRecord
  +-- CodeformerModelRecord
  +-- EsrganModelRecord
  +-- GfpganModelRecord
  +-- SafetyCheckerModelRecord
  +-- VideoGenerationModelRecord
  +-- AudioGenerationModelRecord
  +-- MiscellaneousModelRecord
```

Every record shares these base fields:

| Field                  | Type                              | Description                                     |
| ---------------------- | --------------------------------- | ----------------------------------------------- |
| `name`                 | `str`                             | Model name (also the dict key)                  |
| `description`          | `str \| None`                     | Short description                               |
| `version`              | `str \| None`                     | Model version                                   |
| `record_type`          | `str \| MODEL_REFERENCE_CATEGORY` | Category discriminator                          |
| `model_classification` | `ModelClassification`             | Domain + purpose                                |
| `config`               | `GenericModelRecordConfig`        | Download info                                   |
| `metadata`             | `GenericModelRecordMetadata`      | Timestamps, authorship                          |
| `finetune_series`      | `FineTuneSeriesInfo \| None`      | Fine-tune lineage (e.g., "Pony", "Illustrious") |
| `size_on_disk_bytes`   | `int \| None`                     | Aggregate on-disk size                          |
| `category`             | `MODEL_REFERENCE_CATEGORY \| None`| Normalised category from `record_type`          |

## Key Fields by Category

### Image Generation

| Field                | Type                                     | Description                                                                     |
| -------------------- | ---------------------------------------- | ------------------------------------------------------------------------------- |
| `baseline`           | `KNOWN_IMAGE_GENERATION_BASELINE \| str` | Base architecture (e.g., `stable_diffusion_1`, `stable_diffusion_xl`, `flux_1`) |
| `nsfw`               | `bool`                                   | Whether the model is NSFW                                                       |
| `inpainting`         | `bool \| None`                           | Whether it's an inpainting model                                                |
| `style`              | `MODEL_STYLE \| None`                    | Visual style category                                                           |
| `tags`               | `list[str]`                              | Searchable tags                                                                 |
| `trigger`            | `list[str]`                              | Trigger words for activation                                                    |
| `homepage`           | `str \| None`                            | Link to model homepage                                                          |
| `min_bridge_version` | `int \| None`                            | Minimum AI-Horde-Worker version required                                        |

### Text Generation

| Field              | Type                | Description                                         |
| ------------------ | ------------------- | --------------------------------------------------- |
| `baseline`         | `str \| None`       | Base architecture                                   |
| `parameters_count` | `int`               | Parameter count (aliased from `parameters` in JSON) |
| `nsfw`             | `bool`              | Whether the model is NSFW                           |
| `display_name`     | `str \| None`       | Human-friendly display name                         |
| `instruct_format`  | `str \| None`       | Instruction template (ChatML, Mistral, etc.)        |
| `text_model_group` | `str \| None`       | Base model group for variant grouping               |
| `tags`             | `list[str] \| None` | Searchable tags                                     |

### ControlNet

| Field              | Type                              | Description                  |
| ------------------ | --------------------------------- | ---------------------------- |
| `controlnet_style` | `CONTROLNET_STYLE \| str \| None` | Purpose (canny, depth, etc.) |

### CLIP

| Field             | Type          | Description                 |
| ----------------- | ------------- | --------------------------- |
| `pretrained_name` | `str \| None` | Pretrained model identifier |

## Type Narrowing

`manager.get_model_reference()` returns `dict[str, GenericModelRecord]` because it accepts any
category. For category-specific typed access, use the query API (fully typed per category) or an
`isinstance` check.

### The Query API (Recommended)

`manager.query(category)` returns records already typed to the category's record class:

```python
# Each record is an ImageGenerationModelRecord
for model in manager.query("image_generation").to_list():
    print(f"{model.name}: baseline={model.baseline}, nsfw={model.nsfw}")

# Need a typed dict? Build one from the typed list:
image_models = {model.name: model for model in manager.query("image_generation").to_list()}
```

See [Querying Models](querying_models.md) for the full query reference.

### isinstance Checks

```python
from horde_model_reference.model_reference_records import ImageGenerationModelRecord

model = manager.get_model("image_generation", "some_model_name")

if isinstance(model, ImageGenerationModelRecord):
    print(f"Baseline: {model.baseline}")
    print(f"NSFW: {model.nsfw}")
```

### get_record_type_for_category

Look up the record class for a category programmatically:

```python
from horde_model_reference import get_record_type_for_category

record_class = get_record_type_for_category("image_generation")
# Returns ImageGenerationModelRecord
```

## ModelClassification

Every record has a `model_classification` with `domain` and `purpose`:

```python
model = manager.get_model("image_generation", "some_model")
print(model.model_classification.domain)   # e.g., "image"
print(model.model_classification.purpose)  # e.g., "generation"
```

Domains include `image`, `text`, `video`, `audio`. Purposes include `generation`, `classification`, `upscaling`, `restoration`, `safety`, etc.

## Download Configuration

Model download info lives in `config.download`:

```python
model = manager.get_model("image_generation", "some_model")

for download in model.config.download:
    print(f"File: {download.file_name}")
    print(f"URL: {download.file_url}")
    print(f"SHA256: {download.sha256sum}")
    if download.file_purpose:
        print(f"  Component role: {download.file_purpose}")  # e.g. "vae", "text_encoders"
    if download.size_bytes:
        print(f"  Size: {download.size_bytes:,} bytes")
    if download.known_slow_download:
        print("  (known slow download)")

# Aggregate sizes
if model.declared_total_size_bytes:
    print(f"Total: {model.declared_total_size_bytes:,} bytes")
```

`file_purpose` drives on-disk component routing: a file with purpose `"vae"` or `"text_encoders"`
lands in a sibling folder (`vae/` or `text_encoders/`) rather than beside the primary checkpoint,
matching ComfyUI's component loader conventions. See the
[`on_disk_layout`][horde_model_reference.on_disk_layout] module for the full routing rules.

## Baselines

Image models have a `baseline` field indicating the base architecture. Known baselines are registered as `KNOWN_IMAGE_GENERATION_BASELINE` enum values:

```python
from horde_model_reference import KNOWN_IMAGE_GENERATION_BASELINE

# List all known baselines
for baseline in KNOWN_IMAGE_GENERATION_BASELINE:
    print(baseline.value)
# stable_diffusion_1, stable_diffusion_2, stable_diffusion_xl, flux_1, ...
```

## Serialization

Records are Pydantic models, so standard serialization works:

```python
model = manager.get_model("image_generation", "some_model")

# To dict
data = model.model_dump()

# To JSON string
json_str = model.model_dump_json()

# To dict, excluding unset fields
data = model.model_dump(exclude_unset=True)
```

For bulk serialization of an entire category:

```python
models = manager.get_model_reference("image_generation")
json_dict = ModelReferenceManager.model_reference_to_json_dict_safe(models)
```

## Next

- [Querying Models](querying_models.md) -- filter and aggregate records with the fluent API
