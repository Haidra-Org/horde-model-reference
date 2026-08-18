"""Behavioral guarantees of the packaged text guidance seed generator."""

import json
from pathlib import Path

import scripts.seed_text_guidance as seed
from horde_model_reference.text_guidance import TextGuidanceStatus
from horde_model_reference.text_guidance_store import TextGuidanceStore

_RECORDS: dict[str, dict[str, object]] = {
    "publisher/model-b": {"name": "publisher/model-b", "instruct_format": "ChatML"},
    "publisher/model-a": {"name": "publisher/model-a", "instruct_format": "Alpaca"},
    "publisher/model-c": {"name": "publisher/model-c"},
    "koboldcpp/publisher/model-a": {"name": "koboldcpp/publisher/model-a", "instruct_format": "Alpaca"},
}


def _write_input(path: Path, records: dict[str, dict[str, object]]) -> Path:
    """Write a canonical text reference payload in the given key order."""
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def _run(input_path: Path, output_path: Path) -> None:
    """Generate the bootstrap catalog and Markdown report for one input file."""
    assert (
        seed.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--readme",
                str(output_path.with_suffix(".md")),
                "--timestamp",
                "1700000000",
            ],
        )
        == 0
    )


def test_seed_output_does_not_depend_on_input_ordering(tmp_path: Path) -> None:
    """The review artifact must be reproducible whatever order the source reference iterates in."""
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"

    _run(_write_input(tmp_path / "first_input.json", _RECORDS), first_output)
    reversed_records = dict(reversed(list(_RECORDS.items())))
    _run(_write_input(tmp_path / "second_input.json", reversed_records), second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_output.with_suffix(".md").read_bytes() == second_output.with_suffix(".md").read_bytes()


def test_seed_catalog_bootstraps_a_store_and_resolves_models(tmp_path: Path) -> None:
    """The emitted file loads as packaged bootstrap data and publishes guidance for labeled models."""
    bootstrap_path = tmp_path / "bootstrap"
    bootstrap_path.mkdir()
    _run(_write_input(tmp_path / "input.json", _RECORDS), bootstrap_path / "catalog.json")

    store = TextGuidanceStore(root_path=tmp_path / "runtime", bootstrap_path=bootstrap_path, writable=False)

    assert sorted(profile.profile_id for profile in store.list_profiles()) == ["alpaca", "chatml"]
    published = store.resolve("publisher/model-a", legacy_instruct_format="Alpaca")
    assert published.summary.status is TextGuidanceStatus.PUBLISHED
    assert published.primary_profile is not None
    assert published.primary_profile.profile_id == "alpaca"
    # Backend-prefixed projections are duplicates of a canonical model and must not be assigned.
    assert store.get_assignment("koboldcpp/publisher/model-a") is None
    undocumented = store.resolve("publisher/model-c", legacy_instruct_format=None)
    assert undocumented.summary.status is TextGuidanceStatus.UNDOCUMENTED


def test_seed_timestamps_are_pinned_by_the_caller(tmp_path: Path) -> None:
    """Generated metadata carries the requested stamp so reruns produce identical bytes."""
    output_path = tmp_path / "catalog.json"
    _run(_write_input(tmp_path / "input.json", _RECORDS), output_path)

    catalog = json.loads(output_path.read_text(encoding="utf-8"))

    assert catalog["metadata"]["updated_at"] == 1700000000
    assert catalog["profiles"]["alpaca"]["metadata"]["created_at"] == 1700000000
    assert catalog["assignments"]["publisher/model-a"]["metadata"]["updated_at"] == 1700000000
