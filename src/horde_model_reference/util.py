import re


def model_name_to_showcase_folder_name(model_name: str) -> str:
    """Convert a model name to a lowercase, standardized and sanitized showcase folder name.

    Args:
        model_name (str): The model name to convert.

    Returns:
        str: This is a lowercase, sanitized version of the model name.
    """
    model_name = model_name.lower()
    model_name = model_name.replace("'", "")
    return re.sub(r"[^a-z0-9]", "_", model_name)
