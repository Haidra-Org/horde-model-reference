from pathlib import Path

from .create_env_file_example import TIMESTAMP_LINE_INDEX, generate_env_example


def test_on_disk_env_example_matches_generated() -> None:
    """Verify that the on-disk .env.example file matches the automatically generated one.

    Note that this drops the timestamp line as the time value is variable.
    """
    on_disk_env_example_path = Path(__file__).parent.parent / ".env.example"

    on_disk_env_example_content = on_disk_env_example_path.read_text()
    generated_env_example_content = generate_env_example()

    lines = on_disk_env_example_content.splitlines()
    del lines[TIMESTAMP_LINE_INDEX]
    on_disk_env_example_content = "\n".join(lines)

    lines = generated_env_example_content.splitlines()
    del lines[TIMESTAMP_LINE_INDEX]
    generated_env_example_content = "\n".join(lines)

    assert on_disk_env_example_content == generated_env_example_content
