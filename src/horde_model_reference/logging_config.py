"""Logging configuration utilities for horde-model-reference.

This module provides utilities for configuring loguru logging in a library-friendly way.
By default, the library only logs WARNING and above to avoid flooding consumer applications.

The initial configuration is done in __init__.py when the package is imported.
"""

import sys
from typing import Literal

from loguru import logger

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]


def configure_logger(
    level: LogLevel = "WARNING",
    *,
    format_string: str | None = None,
    colorize: bool = True,
) -> None:
    """Configure the horde-model-reference logger.

    This function allows applications to control logging from horde-model-reference.
    By default, only WARNING and above are logged to prevent noise.

    Args:
        level: The minimum log level to display. One of:
              "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
        format_string: Custom format string for log messages. If None, uses default format.
        colorize: Whether to use colored output (default: True)

    Examples:
        ```python
        from horde_model_reference.logging_config import configure_logger

        # Enable debug logging for troubleshooting
        configure_logger("DEBUG")

        # Enable info logging with custom format
        configure_logger("INFO", format_string="{time} - {message}")

        # Disable all but errors
        configure_logger("ERROR")
        ```

    Note:
        You can also use the environment variable HORDE_MODEL_REFERENCE_LOG_LEVEL
        to set the log level without code changes:

        ```bash
        export HORDE_MODEL_REFERENCE_LOG_LEVEL=DEBUG
        python your_script.py
        ```
    """
    # Remove all existing handlers
    logger.remove()

    # Default format if none provided
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Add new handler with specified configuration
    logger.add(
        sys.stderr,
        level=level,
        format=format_string,
        colorize=colorize,
    )


def disable_logging() -> None:
    """Completely disable all logging from horde-model-reference.

    This is useful when you want complete silence from the library.

    Example:
        ```python
        from horde_model_reference.logging_config import disable_logging

        disable_logging()
        # Now the library will not output any logs
        ```
    """
    logger.remove()


def enable_debug_logging() -> None:
    """Enable DEBUG level logging with detailed format.

    This is a convenience function for debugging and troubleshooting.
    Equivalent to configure_logger("DEBUG").

    Example:
        ```python
        from horde_model_reference.logging_config import enable_debug_logging

        # Enable verbose debug output
        enable_debug_logging()
        ```
    """
    configure_logger("DEBUG")


__all__ = [
    "LogLevel",
    "configure_logger",
    "disable_logging",
    "enable_debug_logging",
]
