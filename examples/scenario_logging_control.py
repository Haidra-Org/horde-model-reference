"""Logging Control Example.

This example demonstrates how to control logging verbosity in horde-model-reference.
By default, the library only logs WARNING and above. You can enable more verbose
logging for debugging purposes.
"""

from horde_model_reference import (
    MODEL_REFERENCE_CATEGORY,
    ModelReferenceManager,
    configure_logger,
    disable_logging,
    enable_debug_logging,
)


def example_1_default_logging() -> None:
    """Demonstrate default logging (WARNING and above only).

    This is the default behavior - quiet operation with only important
    warnings and errors shown.
    """
    print("=" * 70)
    print("Example 1: Default Logging (WARNING and above)")
    print("=" * 70)

    manager = ModelReferenceManager()
    models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)

    print(f"\nFetched {len(models)} image generation models")
    print("Notice: No DEBUG logs were shown, only warnings/errors if any occurred.")
    print()


def example_2_enable_debug() -> None:
    """Demonstrate enabling DEBUG logging for troubleshooting.

    Use this when you need to see what's happening under the hood.
    """
    print("=" * 70)
    print("Example 2: Enable DEBUG Logging")
    print("=" * 70)
    print("Enabling DEBUG level logging...")
    print()

    # Enable verbose debug logging
    enable_debug_logging()

    manager = ModelReferenceManager()
    models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.controlnet)

    print(f"\nFetched {len(models)} ControlNet models")
    print("Notice: You saw detailed DEBUG logs above showing internal operations.")
    print()


def example_3_info_level() -> None:
    """Demonstrate INFO level logging (middle ground).

    Shows important operations without the verbosity of DEBUG.
    """
    print("=" * 70)
    print("Example 3: INFO Level Logging")
    print("=" * 70)
    print("Setting log level to INFO...")
    print()

    # Configure to INFO level (more than WARNING, less than DEBUG)
    configure_logger("INFO")

    manager = ModelReferenceManager()
    models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.clip)

    print(f"\nFetched {len(models)} CLIP models")
    print("Notice: You saw INFO and WARNING logs, but not verbose DEBUG logs.")
    print()


def example_4_silence_all() -> None:
    """Demonstrate complete silence.

    Disable all logging when you want no output at all from the library.
    """
    print("=" * 70)
    print("Example 4: Disable All Logging")
    print("=" * 70)
    print("Disabling all logging...")
    print()

    # Completely silence the library
    disable_logging()

    manager = ModelReferenceManager()
    models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.esrgan)

    print(f"\nFetched {len(models)} ESRGAN models")
    print("Notice: No logs were shown at all, completely silent operation.")
    print()


def example_5_environment_variable() -> None:
    """Demonstrate environment variable control.

    You can also control logging via HORDE_MODEL_REFERENCE_LOG_LEVEL
    environment variable without code changes.
    """
    print("=" * 70)
    print("Example 5: Environment Variable Control")
    print("=" * 70)
    print()
    print("You can control logging via environment variable:")
    print()
    print("  export HORDE_MODEL_REFERENCE_LOG_LEVEL=DEBUG")
    print("  python your_script.py")
    print()
    print("Or for a single command:")
    print()
    print("  HORDE_MODEL_REFERENCE_LOG_LEVEL=DEBUG python your_script.py")
    print()
    print("Available levels: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL")
    print()


def main() -> None:
    """Run all logging control examples."""
    print("\n" + "=" * 70)
    print("HORDE MODEL REFERENCE - LOGGING CONTROL EXAMPLES")
    print("=" * 70)
    print()
    print("This script demonstrates different logging configurations.")
    print("Watch how the output changes with each example.")
    print()

    # Example 1: Default (quiet)
    example_1_default_logging()

    # Example 2: Debug mode
    example_2_enable_debug()

    # Example 3: INFO level
    example_3_info_level()

    # Example 4: Silent mode
    example_4_silence_all()

    # Example 5: Environment variable info
    example_5_environment_variable()

    print("=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print()
    print("Best Practices:")
    print("  - Production code: Use default (WARNING) or disable_logging()")
    print("  - Development: Use INFO level for visibility")
    print("  - Debugging: Use DEBUG or enable_debug_logging()")
    print("  - Environment control: Set HORDE_MODEL_REFERENCE_LOG_LEVEL")
    print()


if __name__ == "__main__":
    main()
