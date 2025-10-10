from pathlib import Path


def dynamically_create_library_markdown_stubs() -> None:
    """Dynamically create markdown stubs for every .py file in the project directory and its subdirectories."""
    project_root = Path(__file__).parent.parent
    code_root = Path(__file__).parent.parent / "src" / "horde_model_reference"

    py_files = list(code_root.glob("**/*.py"))

    sorted_py_files = sorted(py_files, key=lambda x: str(x))

    pyfile_lookup = convert_list_of_paths_to_namespaces(sorted_py_files, project_root)

    code_root_paths = list(code_root.glob("**/*"))
    code_root_paths.append(code_root)
    code_root_paths = [path for path in code_root_paths if path.is_dir()]

    folder_lookup = convert_list_of_paths_to_namespaces(code_root_paths, project_root)

    for folder, _namespace in folder_lookup.items():
        relative_folder = folder.relative_to(project_root)
        if relative_folder.parts[0] == "src":
            relative_folder = Path(*relative_folder.parts[1:])
        relative_folder = "docs" / relative_folder
        relative_folder.mkdir(parents=True, exist_ok=True)

        with open(relative_folder / ".pages", "w", encoding="utf-8") as f:
            if relative_folder.name == "horde_model_reference":
                f.write("title: horde_model_reference Code Reference\n")
            else:
                f.write(f"title: {relative_folder.name}\n")

        files_in_folder = list(folder.glob("*.py"))

        files_in_folder = [file for file in files_in_folder if "__" not in str(file)]

        sorted_files_in_folder = sorted(files_in_folder, key=lambda x: str(x))

        if len(sorted_files_in_folder) == 0:
            continue

        for file in sorted_files_in_folder:
            with open(relative_folder / f"{file.stem}.md", "w", encoding="utf-8") as f:
                f.write(f"# {file.stem}\n")
                file_namespace = pyfile_lookup[file]
                f.write(f"::: {file_namespace}\n")


def convert_list_of_paths_to_namespaces(paths: list[Path], root: Path) -> dict[Path, str]:
    """Convert a list of paths to a lookup dictionary of path to namespace."""
    # Convert path to string, remove everything in the path before "horde_model_reference"
    lookup = {path: str(path).replace(str(root), "") for path in paths}

    # Remove any entry with a dunderscore
    lookup = {key: value for key, value in lookup.items() if "__" not in value}

    # If ends in .py, remove that
    lookup = {key: value[:-3] if value.endswith(".py") else value for key, value in lookup.items()}

    # Replace all slashes with dots
    lookup = {key: value.replace("\\", ".") for key, value in lookup.items()}

    # Unix paths too
    lookup = {key: value.replace("/", ".") for key, value in lookup.items()}

    # Remove the first dot
    lookup = {key: value[1:] if value.startswith(".") else value for key, value in lookup.items()}

    # Purge any empty values
    return {key: value for key, value in lookup.items() if value}


if __name__ == "__main__":
    dynamically_create_library_markdown_stubs()
