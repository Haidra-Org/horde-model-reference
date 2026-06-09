# Filter models for a worker

**Goal:** given the model reference, narrow it down to the models *your* node
should actually serve - by baseline, safety, and how much disk/VRAM the weights
need.

This is the most common consumer task. It uses only the public query API; see
the [Querying Models](../tutorials/querying_models.md) tutorial for the full
surface.

## Pick by baseline

A worker usually supports a fixed set of baselines (architectures). Filter to
just those:

```python
from horde_model_reference import ModelReferenceManager, ImageFields

manager = ModelReferenceManager()

supported = {"stable_diffusion_1", "stable_diffusion_xl"}

servable = (
    manager.query("image_generation")
    .where(ImageFields.baseline.is_in(supported))
    .order_by("name")
    .to_list()
)
print(f"{len(servable)} models match your supported baselines")
```

`.for_baseline("stable_diffusion_xl")` is a shorthand when you only support one.

## Exclude what you can't or won't run

Chain filters to drop NSFW models and anything too large for your hardware.
`size_on_disk_bytes` is the on-disk weight size, a reasonable proxy for VRAM
pressure:

```python
servable = (
    manager.query("image_generation")
    .where(ImageFields.baseline.is_in(supported))
    .exclude_nsfw()                                  # image-specific helper
    .where(ImageFields.size_on_disk_bytes < 7_000_000_000)
    .order_by(ImageFields.size_on_disk_bytes)       # smallest first
    .to_list()
)
```

Filters compose left to right, so you can add or remove a clause without
rewriting the rest.

Comparison filters skip models with unknown values. `size_on_disk_bytes` is `int | None`,
and a `<` / `>` comparison matches only records whose value is set. If you would rather keep
unknown-size models:

```python
.filter(lambda m: m.size_on_disk_bytes is None or m.size_on_disk_bytes < 7_000_000_000)
```

## Turn it into a download list

For a quick list of just the URLs, each record exposes them directly:

```python
for model in servable:
    print(model.name)
    for url in model.all_download_urls:
        print(f"  {url}")
```

To fetch and **verify** weights, iterate `config.download` instead - it carries the file name, URL,
and SHA-256 checksum for each file (some models ship more than one). The `known_slow_download` flag (absent on most models, `bool | None`)
marks mirrors that are reliable but slow, so you can lengthen timeouts or download them last:

```python
for model in servable:
    for f in model.config.download:
        print(f"{model.name}: {f.file_name}")
        print(f"  url:    {f.file_url}")
        print(f"  sha256: {f.sha256sum}")
        if f.known_slow_download:
            print("  (known slow mirror -- expect longer downloads)")
```

Check each downloaded file against `sha256sum` before trusting it. (The reference provides the
metadata; performing the download and the hash check is the worker's job.)

## Next

- [Querying Models](../tutorials/querying_models.md) -- operators, predicates, ordering, pagination
