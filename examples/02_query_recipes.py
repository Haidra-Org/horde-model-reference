"""Query recipes: filter, sort, paginate, and aggregate with the fluent API.

Run with:

    uv run examples/02_query_recipes.py

All query operations run in-memory over already-cached records, so there are
no extra network calls once the data is loaded.
"""

from horde_model_reference import ModelReferenceManager, TextFields


def main() -> None:
    """Demonstrate filtering, distinct values, typed fields, and aggregation."""
    manager = ModelReferenceManager()

    # Keyword equality is the simplest filter. `.exclude_nsfw()` is an
    # image-specific helper that `manager.query("image_generation")` exposes.
    sfw = manager.query("image_generation").exclude_nsfw().order_by("name").limit(5).to_list()
    print(f"First {len(sfw)} SFW image models (by name):")
    for model in sfw:
        print(f"  {model.name}")

    # Distinct values are handy for discovering what data exists.
    baselines = manager.query("image_generation").distinct("baseline")
    print(f"\nDistinct image baselines ({len(baselines)}): {sorted(map(str, baselines))}")

    # Typed field references (ImageFields / TextFields / ...) give IDE
    # autocomplete and support comparison operators.
    big_llms = (
        manager.query("text_generation")
        .where(TextFields.parameters_count > 7_000_000_000)
        .order_by(TextFields.parameters_count, descending=True)
        .limit(5)
        .to_list()
    )
    print(f"\nLargest text models over 7B parameters ({len(big_llms)} shown):")
    for model in big_llms:
        print(f"  {model.name}: {model.parameters_count:,} params")

    # Aggregate across every category at once.
    total = manager.query_all().count()
    print(f"\nTotal models across all categories: {total}")


if __name__ == "__main__":
    main()
