# Querying Model References

The library exposes **two ways to read model records**, and picking the right one is the whole story:

| You want…                                              | Use                                  | You get                          |
| ------------------------------------------------------ | ------------------------------------ | -------------------------------- |
| The raw mapping for a category (lookups, `in`, names)  | `manager.get_model_reference(cat)`   | `dict[str, GenericModelRecord]`  |
| One record by name                                     | `manager.get_model(cat, name)`       | a record (raises if missing)     |
| To filter / sort / aggregate / merge sources           | `manager.query(cat)`                 | a typed `ModelQuery` builder     |

Everything below is about the third row - the fluent query builder. All query operations happen
in-memory over already-cached Pydantic records; there are no extra network calls.

## Direct access (the simple case)

```python
from horde_model_reference import ModelReferenceManager, MODEL_REFERENCE_CATEGORY

manager = ModelReferenceManager()

# The whole mapping for a category
models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
sdxl = models.get("stable_diffusion_xl")          # None if absent
all_names = list(models)                            # just the names

# One record, raising if it does not exist (use get_model_or_none to allow missing)
sdxl = manager.get_model(MODEL_REFERENCE_CATEGORY.image_generation, "stable_diffusion_xl")
```

## The query builder

`manager.query(category)` is the **single entry point** for filtering, sorting, and aggregation.
There are no per-category `query_*` methods - `query()` returns the right typed builder for every
category:

```python
q = manager.query("image_generation")   # -> ImageGenerationQuery (extra image helpers)
q = manager.query("text_generation")    # -> TextModelQuery       (extra text helpers)
q = manager.query("controlnet")         # -> ControlNetQuery       (extra controlnet helpers)
q = manager.query("clip")               # -> ModelQuery[ClipModelRecord]
q = manager.query("esrgan")             # -> ModelQuery[EsrganModelRecord]
```

You may pass either the `MODEL_REFERENCE_CATEGORY` member or its string value - **both narrow the
return type** for static type checkers, so `manager.query("image_generation").exclude_nsfw()` and
`manager.query(MODEL_REFERENCE_CATEGORY.image_generation).exclude_nsfw()` are equivalent and both
type-check.

For a query spanning every category at once, use `manager.query_all()`.

## Filtering with `.where()`

### Keyword equality

```python
results = manager.query("image_generation").where(nsfw=False).to_list()
```

### Django-style comparison operators

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

!!! note "Comparisons are None-safe and exclude missing values"
    The comparison operators (`<`, `<=`, `>`, `>=`, and `__gt`/`__lt`/...) match only records whose
    field is set. A record whose field is `None` never matches a comparison -- it is *excluded*, not
    treated as zero and not kept. (`order_by` likewise sorts `None` values last instead of raising.)
    To filter on presence itself, use `.is_none()` / `.is_not_none()`; to keep missing values
    alongside a comparison, combine them with `|` or drop to `.filter(...)`.

### Typed field references

Import field namespaces for IDE autocomplete and static type checking:

```python
from horde_model_reference import ImageFields, false

results = (
    manager.query("image_generation")
    .where(ImageFields.nsfw == false, ImageFields.baseline == "stable_diffusion_xl")
    .to_list()
)
```

Available field namespaces: `ImageFields`, `TextFields`, `ControlNetFields`, `ClipFields`,
`GenericFields`, `VideoFields`, `AudioFields`, and more. Each provides `FieldRef` attributes matching
the record's fields.

!!! note "Unknown field names raise -- they do not silently return nothing"
    Every field name is validated against the record type before filtering. A typo in a string
    keyword (e.g. `.where(nfsw=False)`) raises a clear error rather than quietly matching zero
    records. The typed field references above catch the mistake even earlier -- at edit time, via
    your IDE and type checker -- so prefer them in code you maintain.

`FieldRef` supports `==`, `!=`, `<`, `<=`, `>`, `>=`, `.contains()`, `.is_in()`, `.is_none()`,
`.is_not_none()`:

```python
from horde_model_reference import TextFields, false

# Text models with 7B+ parameters that are SFW
results = (
    manager.query("text_generation")
    .where(TextFields.parameters_count > 7_000_000_000, TextFields.nsfw == false)
    .to_list()
)
```

### Predicate composition

Combine predicates with `&` (and), `|` (or), `~` (not):

```python
from horde_model_reference import ImageFields, false, true

# SFW models on SDXL OR any inpainting model
pred = (ImageFields.nsfw == false()) & (ImageFields.baseline == "stable_diffusion_xl")
pred_alt = ImageFields.inpainting == true()

results = (
    manager.query("image_generation")
    .where(pred | pred_alt)
    .to_list()
)
```

## Tag filtering

For record types with a `tags` field (image, text, video, audio):

```python
# Models with ANY of these tags
manager.query("image_generation").tags_any(["realistic", "generalist"]).to_list()

# Models with ALL of these tags
manager.query("image_generation").tags_all(["realistic", "generalist"]).to_list()

# Models with NONE of these tags
manager.query("image_generation").tags_none(["nsfw", "anime"]).to_list()
```

## Ordering and pagination

```python
from horde_model_reference import ImageFields

manager.query("image_generation").order_by("name").to_list()                 # ascending
manager.query("image_generation").order_by("name", descending=True).to_list()
manager.query("image_generation").order_by(ImageFields.size_on_disk_bytes.desc()).to_list()

manager.query("image_generation").limit(10).to_list()              # first 10
manager.query("image_generation").offset(5).limit(10).to_list()    # skip 5, take 10
```

## Terminal operations

Every query chain ends with a terminal operation:

| Method             | Returns                   | Description                    |
| ------------------ | ------------------------- | ------------------------------ |
| `.to_list()`       | `list[T]`                 | All matching records           |
| `.first()`         | `T \| None`               | First match, or `None`         |
| `.count()`         | `int`                     | Number of matches              |
| `.distinct(field)` | `list[object]`            | Unique values of a field       |
| `.group_by(field)` | `dict[Hashable, list[T]]` | Records grouped by field value |

```python
count = manager.query("image_generation").exclude_nsfw().count()
baselines = manager.query("image_generation").distinct("baseline")
by_baseline = manager.query("image_generation").group_by("baseline")
```

## Category-specific helpers

The three domain categories return enriched builders with extra chainable helpers.

### Image generation (`ImageGenerationQuery`)

```python
q = manager.query("image_generation")
q.exclude_nsfw()                         # SFW only
q.only_nsfw()                            # NSFW only
q.only_inpainting()                      # Inpainting models only
q.exclude_inpainting()                   # Exclude inpainting
q.for_baseline("stable_diffusion_xl")    # Filter by baseline
```

### Text generation (`TextModelQuery`)

```python
q = manager.query("text_generation")
q.for_backend("koboldcpp")              # Models for a specific backend
q.exclude_backend_variations()          # Remove legacy backend-prefixed duplicates
q.only_quantized()                      # Only quantized variants
q.exclude_quantized()                   # Exclude quantized variants
q.group_by_base_model()                 # Terminal: group variants by base model name
```

### ControlNet (`ControlNetQuery`)

```python
q = manager.query("controlnet")
q.for_style("canny")                    # Only canny-style ControlNets
q.group_by_style()                      # Terminal: group by style
```

## Source-aware queries (third-party providers)

By default every read returns only the **canonical horde data** - the official horde dataset, as
opposed to records from a third-party provider (source id `"horde"`; see the
[Glossary](../reference/glossary.md#canonical-data-the-horde-source)). If you have
[registered third-party providers](registering_providers.md), the `source=` argument selects where
records come from:

```python
from horde_model_reference import ANY_SOURCE  # == "any"

# Canonical only (the default)
manager.query("image_generation").to_list()

# A single provider
manager.query("image_generation", source="civitai").to_list()

# Canonical merged with every registered provider
manager.query("image_generation", source=ANY_SOURCE).to_list()

# An explicit, ordered set - ordering controls collision precedence (earlier wins)
manager.query("image_generation", source=["horde", "civitai"]).to_list()
```

`get_model_reference(cat, source=...)` and `get_model(cat, name, source=...)` accept the same
selector.

### Collisions and provenance

When more than one source is selected, records are de-duplicated by name and the canonical (or
earlier-listed) source wins. The builder lets you inspect provenance and collisions:

```python
q = manager.query("image_generation", source=ANY_SOURCE)

q.has_duplicate_names()      # bool: did any name appear in more than one source?
q.duplicate_names()          # {name: [source_id, ...]} for colliding names (pre-dedup view)
q.where_source("civitai")    # keep only records contributed by a given source
q.sources()                  # source id aligned to each result of the current query
q.to_list_with_source()      # [(record, source_id), ...]
q.group_by_source()          # {source_id: [record, ...]}
```

See [Registering & Consuming Providers](registering_providers.md) for how to register a source.

## Cross-category queries

```python
total = manager.query_all().count()
results = manager.query_all().filter(lambda r: "flux" in r.name.lower()).to_list()
```

## Arbitrary predicates

Use `.filter()` for logic that doesn't fit the built-in operators:

```python
results = (
    manager.query("image_generation")
    .filter(lambda r: len(r.config.download) >= 2)
    .to_list()
)
```

## Worked example

**Find the 5 largest SFW SDXL inpainting models:**

```python
from horde_model_reference import ModelReferenceManager, ImageFields, false, true

manager = ModelReferenceManager()

results = (
    manager.query("image_generation")
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

## Next steps

- [Registering & Consuming Providers](registering_providers.md) -- Contribute models from a third-party source
- [Working with Records](working_with_records.md) -- Understand record types, fields, and serialization
- [Configuration & Troubleshooting](configuration_and_troubleshooting.md) -- Env vars, debugging, common issues
```
