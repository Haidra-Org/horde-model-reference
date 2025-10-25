import argparse
import sys

from loguru import logger

logger.remove()

from horde_model_reference import MODEL_REFERENCE_CATEGORY, ModelReferenceManager  # noqa: E402

ALIASES = {"stable_diffusion": "image_generation"}


def configure_logger(quiet: bool) -> None:
    """Configure the loguru logger sink based on quiet flag."""
    logger.remove()
    if quiet:
        logger.add(lambda _: None)
    else:
        logger.add(lambda msg: print(msg, end=""))


def get_all_names(model_reference_category: MODEL_REFERENCE_CATEGORY, refresh: bool) -> list[str]:
    """Get all model names for a given model reference category."""
    logger.debug(f"Getting all names for category: {model_reference_category}")
    model_reference_manager = ModelReferenceManager(lazy_mode=False)
    all_references = model_reference_manager.get_all_model_references(overwrite_existing=refresh)

    if model_reference_category not in all_references:
        logger.warning(f"No references found for category: {model_reference_category}")
        return []

    return list(all_references[model_reference_category].keys())


def main() -> None:
    """Script entry point to get all model names for a given category."""
    parser = argparse.ArgumentParser(description="Get all model names for a given category.")
    parser.add_argument(
        "category",
        type=str,
        help="The model reference category to query (e.g., 'image_generation', 'text_generation').",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="If set, enables all logs.",
    )

    parser.add_argument(
        "-r",
        "--refresh",
        action="store_true",
        help="If set, refreshes the model references from the backend.",
    )
    args = parser.parse_args()

    configure_logger(not args.debug)

    # Normalize aliases (e.g., stable_diffusion -> image_generation)
    category_str = ALIASES.get(args.category, args.category)
    if category_str != args.category:
        logger.info(f"Mapping '{args.category}' to '{category_str}' category.")

    try:
        category_enum = MODEL_REFERENCE_CATEGORY(category_str)
    except ValueError:
        logger.error(
            f"Invalid category: {category_str}. Valid categories are: {[c.value for c in MODEL_REFERENCE_CATEGORY]}",
        )
        sys.exit(1)

    names = get_all_names(category_enum, args.refresh)

    if names:
        logger.info(f"Model names in category '{category_str}':")
        for name in names:
            print(f"{name}")
    else:
        logger.info(f"No model names found in category '{category_str}'.")


if __name__ == "__main__":
    main()
