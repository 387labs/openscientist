"""Migrate legacy raw-dict transcripts to the typed `TranscriptEntry` shape.

Walks every ``jobs/<uuid>/provenance/iter*_transcript.json`` and
``jobs/<uuid>/provenance/report_transcript.json`` under the given
jobs directory, detects files still in the pre-PR-8 raw SDK dict
shape, runs ``CLAUDE.deserialize`` on them, and rewrites in place
via ``save_transcript``. The original bytes are preserved alongside
as ``*.legacy.json`` before any overwrite.

Idempotent: files already in the typed shape are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from openscientist.transcript import (
    CLAUDE,
    TranscriptAdapter,
    TranscriptEntry,
    save_transcript,
)

logger = logging.getLogger("migrate_legacy_transcripts")


class _MigrationOutcome:
    """Result of attempting to migrate a single file."""

    ALREADY_TYPED = "already_typed"
    MIGRATED = "migrated"
    UNREADABLE = "unreadable"
    UNRECOGNISED = "unrecognised"
    BACKUP_EXISTS = "backup_exists"


def _classify_and_migrate(path: Path, *, dry_run: bool) -> str:
    """Inspect ``path`` and migrate if it holds a raw-dict transcript."""
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s as JSON: %s", path, exc)
        return _MigrationOutcome.UNREADABLE

    try:
        TranscriptAdapter.validate_python(raw)
        return _MigrationOutcome.ALREADY_TYPED
    except ValidationError:
        pass

    if not isinstance(raw, list):
        logger.warning("%s is not a JSON list, cannot migrate", path)
        return _MigrationOutcome.UNRECOGNISED

    try:
        entries: list[TranscriptEntry] = CLAUDE.deserialize(raw)
    except (TypeError, ValidationError) as exc:
        logger.warning("%s does not match the Claude raw shape: %s", path, exc)
        return _MigrationOutcome.UNRECOGNISED

    backup = path.with_suffix(path.suffix + ".legacy.json")
    if backup.exists():
        logger.warning(
            "%s has an existing %s sibling, refusing to overwrite. "
            "Resolve the conflict by hand and re-run.",
            path,
            backup.name,
        )
        return _MigrationOutcome.BACKUP_EXISTS
    if dry_run:
        logger.info("[dry-run] would migrate %s (backup -> %s)", path, backup.name)
    else:
        backup.write_bytes(path.read_bytes())
        save_transcript(path, entries)
        logger.info("Migrated %s (backup -> %s)", path, backup.name)
    return _MigrationOutcome.MIGRATED


def _iter_provenance_transcripts(jobs_dir: Path) -> list[Path]:
    """Return every ``iter*_transcript.json`` and ``report_transcript.json``
    under ``jobs_dir/<uuid>/provenance/``."""
    paths: set[Path] = set()
    paths.update(jobs_dir.glob("*/provenance/iter*_transcript.json"))
    paths.update(jobs_dir.glob("*/provenance/report_transcript.json"))
    return sorted(paths)


def migrate_jobs_dir(jobs_dir: Path, *, dry_run: bool = False) -> dict[str, int]:
    """Migrate every legacy transcript under ``jobs_dir`` and return counts.

    Returns a dict mapping outcome label to the number of files in
    that bucket.
    """
    counts = {
        _MigrationOutcome.ALREADY_TYPED: 0,
        _MigrationOutcome.MIGRATED: 0,
        _MigrationOutcome.UNREADABLE: 0,
        _MigrationOutcome.UNRECOGNISED: 0,
        _MigrationOutcome.BACKUP_EXISTS: 0,
    }
    for path in _iter_provenance_transcripts(jobs_dir):
        outcome = _classify_and_migrate(path, dry_run=dry_run)
        counts[outcome] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path("jobs"),
        help="Path to the jobs directory (default: ./jobs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be migrated without writing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable INFO-level logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if not args.jobs_dir.is_dir():
        print(f"jobs-dir does not exist: {args.jobs_dir}", file=sys.stderr)
        return 2

    counts = migrate_jobs_dir(args.jobs_dir, dry_run=args.dry_run)
    print(
        f"already typed: {counts[_MigrationOutcome.ALREADY_TYPED]}, "
        f"migrated: {counts[_MigrationOutcome.MIGRATED]}, "
        f"unreadable: {counts[_MigrationOutcome.UNREADABLE]}, "
        f"unrecognised: {counts[_MigrationOutcome.UNRECOGNISED]}, "
        f"backup exists: {counts[_MigrationOutcome.BACKUP_EXISTS]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
