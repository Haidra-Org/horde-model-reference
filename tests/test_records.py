from horde_model_reference.model_reference_records import DownloadRecord, StableDiffusion_ModelRecord


def test_stable_diffusion_model_record():
    """Tests the StableDiffusion_ModelRecord class."""
    # Create a record
    StableDiffusion_ModelRecord(
        name="test_name",
        description="test_description",
        version="test_version",
        style="test_style",
        purpose="test_purpose",
        inpainting=False,
        baseline="test_baseline",
        tags=["test_tag"],
        nsfw=False,
        config={
            "test_config": [
                DownloadRecord(file_name="test_file_name", file_url="test_file_url", sha256sum="test_sha256sum"),
            ],
        },
    )
