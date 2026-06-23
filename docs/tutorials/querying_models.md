# Querying Model References

The library provides two ways to read model records:

| You want                                              | Use                                | You get                         |
| ----------------------------------------------------- | ---------------------------------- | ------------------------------- |
| The raw mapping for a category (lookups, `in`, names) | `manager.get_model_reference(cat)` | `dict[str, GenericModelRecord]` |
| One record by name                                    | `manager.get_model(cat, name)`     | a record (raises if missing)    |
| To filter / sort / aggregate / merge sources          | `manager.query(cat)`               | a typed `ModelQuery` builder    |

This page covers the query builder. All query operations run in-memory over already-cached Pydantic records; there are no extra network calls.

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

### Typed field references (recommended)

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
`GenericFields`, `VideoFields`, `AudioFields`, `BlipFields`, `CodeformerFields`, `EsrganFields`,
`GfpganFields`, `SafetyCheckerFields`, and `MiscellaneousFields`. Each provides `FieldRef` attributes
matching the record's fields.

Field references catch typos at edit time through your IDE and type checker, while string
keywords raise a clear error at runtime rather than silently matching zero records. For this
reason, prefer typed field references in code you maintain.

Comparisons with `FieldRef` operators (`<`, `<=`, `>`, `>=`) are None-safe: a record whose field is
`None` is excluded from the comparison, not treated as zero. Use `.is_none()` / `.is_not_none()` to
filter on presence itself. `order_by` likewise sorts `None` values last.

`FieldRef` supports `==`, `!=`, `<`, `<=`, `>`, `>=`, `.contains()`, `.is_in()`, `.is_none()`,
`.is_not_none()`, `.is_true()`, and `.is_false()`. For boolean fields, prefer `.is_true()` and
`.is_false()`:

```python
from horde_model_reference import ImageFields

# Boolean filter with is_true() / is_false()
results = (
    manager.query("image_generation")
    .where(ImageFields.nsfw.is_false(), ImageFields.inpainting.is_true())
    .to_list()
)
```

For equality comparisons with non-boolean values, use `==` directly. For boolean fields, an
alternative is `false()` / `true()` (imported from `horde_model_reference`) — these produce
`FieldRef` predicates instead of comparing against Python's built-in `False` / `True`:

```python
from horde_model_reference import ImageFields, false

# Equivalent to ImageFields.nsfw.is_false()
results = (
    manager.query("image_generation")
    .where(ImageFields.nsfw == false)
    .to_list()
)
```

```python
from horde_model_reference import TextFields, false

# Text models with 7B+ parameters that are SFW
results = (
    manager.query("text_generation")
    .where(TextFields.parameters_count > 7_000_000_000, TextFields.nsfw == false)
    .to_list()
)
```

#### Keyword strings (alternative)

As an alternative to typed field references, you can pass field names as string keywords. Append
`__gt`, `__lt`, `__gte`, `__lte`, `__ne`, `__in`, or `__contains` for comparisons:

```python
large = manager.query("image_generation").where(size_on_disk_bytes__gt=1_000_000_000).to_list()
```

String keywords validate field names at runtime (typos raise an error rather than silently
matching nothing), but misspellings are only caught when the code runs. Typed field references
catch them in your IDE instead, so prefer `FieldRef` in code you maintain.

## Predicate composition

Combine predicates with `&` (and), `|` (or), `~` (not):

```python
from horde_model_reference import ImageFields

# SFW models on SDXL OR any inpainting model
pred = ImageFields.nsfw.is_false() & (ImageFields.baseline == "stable_diffusion_xl")
pred_alt = ImageFields.inpainting.is_true()

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

Every query chain ends with a terminal: `.to_list()` (all matches), `.first()` (first or
`None`), `.count()`, `.distinct(field)`, and `.group_by(field)`.

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

By default every read returns only the canonical horde data (the official dataset, source id
`"horde"`). If you have [registered third-party providers](registering_providers.md), the
`source=` argument selects where records come from:

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

When more than one source is selected, records are de-duplicated by name and the **first source
in selector order** wins. The canonical source keeps its position wherever it appears in the
selector, so `["pending", "horde"]` lets a pending beta override canonical while the default
`["horde"]` is canonical-only. The builder lets you inspect provenance and collisions:

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
from horde_model_reference import ModelReferenceManager, ImageFields

manager = ModelReferenceManager()

results = (
    manager.query("image_generation")
    .where(
        ImageFields.nsfw.is_false(),
        ImageFields.baseline == "stable_diffusion_xl",
        ImageFields.inpainting.is_true(),
    )
    .order_by(ImageFields.size_on_disk_bytes.desc())
    .limit(5)
    .to_list()
)

for model in results:
    size_mb = (model.size_on_disk_bytes or 0) / 1_000_000
    print(f"{model.name}: {size_mb:.0f} MB")
```

## Next

- [Working with Records](working_with_records.md) -- record types, fields, and serialization
- [Registering & Consuming Providers](registering_providers.md) -- contribute models from third-party sources
```
