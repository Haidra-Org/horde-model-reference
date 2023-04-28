import re


def model_name_to_showcase_folder_name(model_name: str) -> str:
    model_name = model_name.lower()
    model_name = model_name.replace("'", "")
    model_name = re.sub(r"[^a-z0-9]", "_", model_name)
    return model_name
