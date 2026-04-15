# Querying Model References

The library provides a fluent query API that lets you filter, sort, and aggregate model records with a chainable, type-safe interface. All operations happen in-memory over already-cached Pydantic models -- no extra network calls.

## Basic Access vs. Query Builder

**Dict access** works for simple lookups:

```python
from horde_model_reference import ModelReferenceManager, MODEL_REFERENCE_CATEGORY

manager = ModelReferenceManager()
models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)

# Direct lookup by name
sdxl = models.get("stable_diffusion_xl")
```

**The query builder** is better when you need filtering, sorting, or aggregation:

```python
sfw_sdxl_models = (
    manager.query("image_generation")
    .where(nsfw=False, baseline="stable_diffusion_xl")
    .order_by("name")
    .to_list()
)
```

## Starting a Query

Use `manager.query(category)` for any category. Typed shortcuts exist for common categories:

```python
# Generic -- works for any category
q = manager.query("image_generation")

# Typed shortcuts -- return specialized query builders with extra helpers
q = manager.query_image_generation()   # ImageGenerationQuery
q = manager.query_text_generation()    # TextModelQuery
q = manager.query_controlnet()         # ControlNetQuery

# Other categories
q = manager.query_clip()
q = manager.query_esrgan()
q = manager.query_blip()

# Cross-category (all models from all categories)
q = manager.query_all()
```

## Filtering with `.where()`

### Keyword Equality

```python
results = manager.query("image_generation").where(nsfw=False).to_list()
```

### Django-Style Comparison Operators

Append `__operator` to the field name:

| Operator     | Meaning                         |
| ------------ | ------------------------------- |
| `__gt`       | Greater than                    |
| `__gte`      | Greater than or equal           |
| `__lt`       | Less than                       |
| `__lte`      | Less than or equal              |
| `__ne`       | Not equal                       |
| `__in`       | Value is in a collection        |
| `__contains` | Collection field contains value |

```python
# Models larger than 1 GB
large = (
    manager.query("image_generation")
    .where(size_on_disk_bytes__gt=1_000_000_000)
    .to_list()
)

# Text models with more than 7 billion parameters
big_llms = (
    manager.query("text_generation")
    .where(parameters_count__gt=7_000_000_000)
    .to_list()
)
```

### Typed Field References

Import field namespaces for IDE autocomplete and static type checking:

```python
from horde_model_reference import ImageFields, false

results = (
    manager.query_image_generation()
    .where(ImageFields.nsfw == false, ImageFields.baseline == "stable_diffusion_xl")
    .to_list()
)
```

Available field namespaces: `ImageFields`, `TextFields`, `ControlNetFields`, `ClipFields`, `GenericFields`, `VideoFields`, `AudioFields`, and more. Each provides `FieldRef` attributes matching the record's fields.

`FieldRef` supports `==`, `!=`, `<`, `<=`, `>`, `>=`, `.contains()`, `.is_in()`, `.is_none()`, `.is_not_none()`:

```python
from horde_model_reference import TextFields

# Text models with 7B+ parameters that are SFW
results = (
    manager.query_text_generation()
    .where(TextFields.parameters_count > 7_000_000_000, TextFields.nsfw == false)
    .to_list()
)
```

### Predicate Composition

Combine predicates with `&` (and), `|` (or), `~` (not):

```python
from horde_model_reference import ImageFields, false, true

# SFW models on SDXL OR any inpainting model
pred = (ImageFields.nsfw == false()) & (ImageFields.baseline == "stable_diffusion_xl")
pred_alt = ImageFields.inpainting == true()

results = (
    manager.query_image_generation()
    .where(pred | pred_alt)
    .to_list()
)
```

## Tag Filtering

For record types with a `tags` field (image, text, video, audio):

```python
# Models with ANY of these tags
manager.query_image_generation().tags_any(["realistic", "generalist"]).to_list()

# Models with ALL of these tags
manager.query_image_generation().tags_all(["realistic", "generalist"]).to_list()

# Models with NONE of these tags
manager.query_image_generation().tags_none(["nsfw", "anime"]).to_list()
```

## Ordering and Pagination

### Sorting

```python
# Ascending (default)
manager.query("image_generation").order_by("name").to_list()

# Descending
manager.query("image_generation").order_by("name", descending=True).to_list()

# Using field refs
from horde_model_reference import ImageFields
manager.query_image_generation().order_by(ImageFields.size_on_disk_bytes.desc()).to_list()
```

### Pagination

```python
# First 10 results
manager.query("image_generation").limit(10).to_list()

# Skip first 5, take next 10
manager.query("image_generation").offset(5).limit(10).to_list()
```

## Terminal Operations

Every query chain ends with a terminal operation:

| Method             | Returns                   | Description                    |
| ------------------ | ------------------------- | ------------------------------ |
| `.to_list()`       | `list[T]`                 | All matching records           |
| `.first()`         | `T \| None`               | First match, or `None`         |
| `.count()`         | `int`                     | Number of matches              |
| `.distinct(field)` | `list[object]`            | Unique values of a field       |
| `.group_by(field)` | `dict[Hashable, list[T]]` | Records grouped by field value |

```python
# How many SFW image models?
count = manager.query_image_generation().exclude_nsfw().count()

# What baselines are in use?
baselines = manager.query_image_generation().distinct("baseline")

# Group by baseline
by_baseline = manager.query_image_generation().group_by("baseline")
for baseline, models in by_baseline.items():
    print(f"{baseline}: {len(models)} models")
```

## Category-Specific Builders

### Image Generation

`ImageGenerationQuery` adds convenience methods:

```python
q = manager.query_image_generation()

q.exclude_nsfw()                         # SFW only
q.only_nsfw()                            # NSFW only
q.only_inpainting()                      # Inpainting models only
q.exclude_inpainting()                   # Exclude inpainting
q.for_baseline("stable_diffusion_xl")    # Filter by baseline
```

### Text Generation

`TextModelQuery` adds text-specific helpers:

```python
q = manager.query_text_generation()

q.for_backend("koboldcpp")              # Models for a specific backend
q.exclude_backend_variations()          # Remove legacy backend-prefixed duplicates
q.only_quantized()                      # Only quantized variants
q.exclude_quantized()                   # Exclude quantized variants
q.group_by_base_model()                 # Terminal: group variants by base model name
```

### ControlNet

`ControlNetQuery` adds style filtering:

```python
q = manager.query_controlnet()

q.for_style("canny")                    # Only canny-style ControlNets
q.group_by_style()                      # Terminal: group by style
```

## Cross-Category Queries

Query across all categories at once:

```python
# Count all models in the entire reference
total = manager.query_all().count()

# Find all models matching a name pattern
results = manager.query_all().filter(lambda r: "flux" in r.name.lower()).to_list()
```

## Arbitrary Predicates

Use `.filter()` for logic that doesn't fit the built-in operators:

```python
# Models with at least 2 download files
results = (
    manager.query("image_generation")
    .filter(lambda r: len(r.config.download) >= 2)
    .to_list()
)
```

## Worked Example

**Find the 5 largest SFW SDXL inpainting models:**

```python
from horde_model_reference import ModelReferenceManager, ImageFields, false, true

manager = ModelReferenceManager()

results = (
    manager.query_image_generation()
    .where(
        ImageFields.nsfw == false(),
        ImageFields.baseline == "stable_diffusion_xl",
        ImageFields.inpainting == true(),
    )
    .order_by(ImageFields.size_on_disk_bytes.desc())
    .limit(5)
    .to_list()
)

for model in results:
    size_mb = (model.size_on_disk_bytes or 0) / 1_000_000
    print(f"{model.name}: {size_mb:.0f} MB")
```

## Next Steps

- [Working with Records](working_with_records.md) -- Understand record types, fields, and serialization
- [Configuration & Troubleshooting](configuration_and_troubleshooting.md) -- Env vars, debugging, common issues
