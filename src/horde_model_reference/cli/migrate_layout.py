"""One-time, symlink-safe migration of an on-disk model tree to the canonical layout.

The canonical on-disk layout is defined by :mod:`horde_model_reference.on_disk_layout`: each model file
sits under its category folder, except component files (a VAE, text encoders) which route to a sibling
folder. This tool relocates any declared file that is sitting in the flat category folder when the registry
now routes it to a sibling, so an older tree matches what the loaders and the download engine expect.

It is deliberately conservative:
    - Dry-run by default; nothing moves unless explicitly applied.
    - Never clobbers: a file already at its canonical path is left alone (so the tool is idempotent).
    - Symlink-safe: a source that is a symlink, or sits under a symlinked ancestor, is reported and skipped
      so user-defined symlink layouts are never disturbed.
    - Sidecars (``.sha256``/``.md5``) travel with their file.

Because every category's folder name is unchanged from prior releases, this is a verified no-op on standard
installs; it exists to handle genuine relocations and component files that an older tree placed flat.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from horde_model_reference.on_disk_layout import category_folder, component_relative_path, resolve_weights_root

if TYPE_CHECKING:
    from collections.abc import Mapping

    from horde_model_reference.model_reference_records import GenericModelRecord

__all__ = [
    "MigrationPlan",
    "PlannedMove",
    "apply_layout_migration",
    "main",
    "plan_layout_migration",
]

_SIDECAR_SUFFIXES = (".sha256", ".md5")


@dataclass(frozen=True)
class PlannedMove:
    """A single file relocation the migration would perform."""

    source: Path
    """The current (flat) on-disk path of the file."""
    destination: Path
    """The canonical on-disk path the file should move to."""
    sidecars: tuple[tuple[Path, Path], ...] = ()
    """``(source, destination)`` pairs for any ``.sha256``/``.md5`` sidecars that accompany the file."""


@dataclass(frozen=True)
class MigrationPlan:
    """The set of moves (and skips) a migration run would perform."""

    moves: list[PlannedMove]
    skipped_symlinks: list[Path]
    """Sources skipped because they are a symlink or sit under a symlinked ancestor."""

    @property
    def is_noop(self) -> bool:
        """Return whether the plan would change nothing on disk."""
        return not self.moves


def _under_symlink(path: Path, root: Path) -> bool:
    """Return whether *path* itself, or any ancestor below *root*, is a symlink."""
    if path.is_symlink():
        return True
    for ancestor in path.parents:
        if ancestor == root:
            break
        if ancestor.is_symlink():
            return True
    return False


def plan_layout_migration(reference: Mapping[str, GenericModelRecord], root: Path) -> MigrationPlan:
    """Plan the moves needed to bring *root* into the canonical layout for *reference*.

    Args:
        reference: The model records (name -> record) whose files should be checked.
        root: The model-weights root under which the category folders live.

    Returns:
        A :class:`MigrationPlan`; :attr:`MigrationPlan.is_noop` is True when nothing needs moving.
    """
    root = Path(root)
    moves: list[PlannedMove] = []
    skipped: list[Path] = []

    for record in reference.values():
        category = record.category
        if category is None:
            continue
        folder = category_folder(category)
        if folder is None:
            continue
        category_dir = root / folder
        for download in record.config.download:
            relative = component_relative_path(download.file_name, download.file_purpose)
            canonical = Path(os.path.normpath(category_dir / relative))
            flat = Path(os.path.normpath(category_dir / download.file_name))
            if canonical == flat:
                continue  # not a routed component; its flat location is already canonical
            if canonical.exists():
                continue  # already migrated (idempotent, and we never clobber)
            if not flat.exists():
                continue  # nothing on disk to relocate
            if _under_symlink(flat, root):
                skipped.append(flat)
                continue
            sidecars = tuple(
                (flat.with_suffix(suffix), canonical.with_suffix(suffix))
                for suffix in _SIDECAR_SUFFIXES
                if flat.with_suffix(suffix).exists()
            )
            moves.append(PlannedMove(source=flat, destination=canonical, sidecars=sidecars))

    return MigrationPlan(moves=moves, skipped_symlinks=skipped)


def apply_layout_migration(plan: MigrationPlan) -> list[PlannedMove]:
    """Execute *plan*'s moves, skipping any whose destination has since appeared. Returns the moves done."""
    applied: list[PlannedMove] = []
    for move in plan.moves:
        if move.destination.exists():
            logger.warning("Skipping move; destination already exists: {}", move.destination)
            continue
        move.destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(move.source, move.destination)
        for sidecar_source, sidecar_destination in move.sidecars:
            if sidecar_source.exists() and not sidecar_destination.exists():
                sidecar_destination.parent.mkdir(parents=True, exist_ok=True)
                os.rename(sidecar_source, sidecar_destination)
        applied.append(move)
    return applied


def _load_all_references() -> dict[str, GenericModelRecord]:
    """Load every available category's reference into one mapping, tolerating missing categories."""
    from horde_model_reference.model_reference_manager import ModelReferenceManager

    manager = ModelReferenceManager.get_instance() if ModelReferenceManager.has_instance() else ModelReferenceManager()
    combined: dict[str, GenericModelRecord] = {}
    for records in manager.get_all_model_references_or_none(safe_mode=True).values():
        if records:
            combined.update(records)
    return combined


def report_plan(plan: MigrationPlan, root: Path, *, apply: bool) -> None:
    """Log a human-readable summary of *plan* (used by the console script)."""
    for move in plan.moves:
        logger.info("{} {} -> {}", "MOVE" if apply else "WOULD MOVE", move.source, move.destination)
    for skipped in plan.skipped_symlinks:
        logger.warning("Skipped (symlink or under a symlinked ancestor): {}", skipped)
    if plan.is_noop and not plan.skipped_symlinks:
        logger.info("Model layout already current; nothing to migrate (root={})", root)


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``migrate-model-layout``."""
    parser = argparse.ArgumentParser(
        prog="migrate-model-layout",
        description="Migrate an on-disk model tree to the canonical horde_model_reference layout.",
    )
    parser.add_argument("--apply", action="store_true", help="Perform the moves (default: dry-run).")
    parser.add_argument(
        "--cache-home",
        default=None,
        help="Model cache home (defaults to the AIWORKER_CACHE_HOME environment variable).",
    )
    args = parser.parse_args(argv)

    root = resolve_weights_root(args.cache_home)
    reference = _load_all_references()
    plan = plan_layout_migration(reference, root)
    report_plan(plan, root, apply=args.apply)

    if args.apply:
        applied = apply_layout_migration(plan)
        logger.info("Applied {} move(s).", len(applied))
    elif plan.moves:
        logger.info("Dry run: {} move(s) would be applied. Re-run with --apply to perform them.", len(plan.moves))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
