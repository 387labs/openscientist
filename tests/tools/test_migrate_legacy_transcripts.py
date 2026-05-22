"""Tests for `tools/migrate_legacy_transcripts.py`."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from openscientist.transcript import (
    CLAUDE,
    AssistantText,
    TranscriptAdapter,
    load_transcript,
    save_transcript,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATOR_PATH = REPO_ROOT / "tools" / "migrate_legacy_transcripts.py"
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "transcripts"
    / "claude_agent_sdk"
    / "capital_of_france_iter04.json.gz"
)


def _load_migrator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("migrate_legacy_transcripts", MIGRATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def migrator() -> ModuleType:
    return _load_migrator()


@pytest.fixture
def legacy_raw() -> list[dict[str, Any]]:
    """A real raw-dict Claude transcript captured before PR 8."""
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        loaded: list[dict[str, Any]] = json.load(f)
    return loaded


def _make_job(jobs_dir: Path, name: str, transcripts: dict[str, str]) -> Path:
    """Create a ``jobs/<name>/provenance/`` dir populated with the
    given ``{filename: raw_json_string}`` mapping."""
    provenance = jobs_dir / name / "provenance"
    provenance.mkdir(parents=True)
    for filename, body in transcripts.items():
        (provenance / filename).write_text(body)
    return provenance


def test_migrates_legacy_iteration_transcript(
    tmp_path: Path, migrator: ModuleType, legacy_raw: list[dict[str, Any]]
) -> None:
    jobs_dir = tmp_path / "jobs"
    provenance = _make_job(
        jobs_dir,
        "job-a",
        {"iter1_transcript.json": json.dumps(legacy_raw)},
    )
    target = provenance / "iter1_transcript.json"

    counts = migrator.migrate_jobs_dir(jobs_dir)

    assert counts[migrator._MigrationOutcome.MIGRATED] == 1
    assert counts[migrator._MigrationOutcome.ALREADY_TYPED] == 0
    typed = load_transcript(target)
    assert typed == CLAUDE.deserialize(legacy_raw)

    backup = target.with_suffix(target.suffix + ".legacy.json")
    assert backup.is_file()
    assert json.loads(backup.read_text()) == legacy_raw


def test_migrates_report_transcript(
    tmp_path: Path, migrator: ModuleType, legacy_raw: list[dict[str, Any]]
) -> None:
    jobs_dir = tmp_path / "jobs"
    _make_job(
        jobs_dir,
        "job-b",
        {"report_transcript.json": json.dumps(legacy_raw)},
    )
    counts = migrator.migrate_jobs_dir(jobs_dir)
    assert counts[migrator._MigrationOutcome.MIGRATED] == 1


def test_already_typed_file_is_left_alone(tmp_path: Path, migrator: ModuleType) -> None:
    jobs_dir = tmp_path / "jobs"
    provenance = jobs_dir / "job-c" / "provenance"
    provenance.mkdir(parents=True)
    target = provenance / "iter1_transcript.json"
    save_transcript(target, [AssistantText(text="hello")])
    before = target.read_bytes()

    counts = migrator.migrate_jobs_dir(jobs_dir)

    assert counts[migrator._MigrationOutcome.ALREADY_TYPED] == 1
    assert counts[migrator._MigrationOutcome.MIGRATED] == 0
    assert target.read_bytes() == before
    assert not target.with_suffix(target.suffix + ".legacy.json").exists()


def test_migration_is_idempotent(
    tmp_path: Path, migrator: ModuleType, legacy_raw: list[dict[str, Any]]
) -> None:
    jobs_dir = tmp_path / "jobs"
    provenance = _make_job(
        jobs_dir,
        "job-d",
        {"iter1_transcript.json": json.dumps(legacy_raw)},
    )
    target = provenance / "iter1_transcript.json"

    first = migrator.migrate_jobs_dir(jobs_dir)
    assert first[migrator._MigrationOutcome.MIGRATED] == 1
    first_typed_bytes = target.read_bytes()

    second = migrator.migrate_jobs_dir(jobs_dir)
    assert second[migrator._MigrationOutcome.MIGRATED] == 0
    assert second[migrator._MigrationOutcome.ALREADY_TYPED] == 1
    assert target.read_bytes() == first_typed_bytes


def test_dry_run_does_not_write(
    tmp_path: Path, migrator: ModuleType, legacy_raw: list[dict[str, Any]]
) -> None:
    jobs_dir = tmp_path / "jobs"
    provenance = _make_job(
        jobs_dir,
        "job-e",
        {"iter1_transcript.json": json.dumps(legacy_raw)},
    )
    target = provenance / "iter1_transcript.json"
    before = target.read_bytes()

    counts = migrator.migrate_jobs_dir(jobs_dir, dry_run=True)

    assert counts[migrator._MigrationOutcome.MIGRATED] == 1
    assert target.read_bytes() == before
    assert not target.with_suffix(target.suffix + ".legacy.json").exists()


def test_unrecognised_garbage_file_is_left_alone(tmp_path: Path, migrator: ModuleType) -> None:
    jobs_dir = tmp_path / "jobs"
    provenance = jobs_dir / "job-f" / "provenance"
    provenance.mkdir(parents=True)
    target = provenance / "iter1_transcript.json"
    target.write_text(json.dumps({"not": "a list"}))
    before = target.read_bytes()

    counts = migrator.migrate_jobs_dir(jobs_dir)

    assert counts[migrator._MigrationOutcome.UNRECOGNISED] == 1
    assert counts[migrator._MigrationOutcome.MIGRATED] == 0
    assert target.read_bytes() == before


def test_skips_files_outside_provenance_dirs(
    tmp_path: Path, migrator: ModuleType, legacy_raw: list[dict[str, Any]]
) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "job-g"
    job_dir.mkdir(parents=True)
    stray = job_dir / "iter1_transcript.json"
    stray.write_text(json.dumps(legacy_raw))

    counts = migrator.migrate_jobs_dir(jobs_dir)

    assert all(v == 0 for v in counts.values())
    assert stray.read_text() == json.dumps(legacy_raw)


def test_roundtrip_matches_pr8_pipeline(
    tmp_path: Path, migrator: ModuleType, legacy_raw: list[dict[str, Any]]
) -> None:
    """A migrated file is byte-identical to what PR 8's pipeline writes."""
    jobs_dir = tmp_path / "jobs"
    provenance = _make_job(
        jobs_dir,
        "job-h",
        {"iter1_transcript.json": json.dumps(legacy_raw)},
    )
    migrated_path = provenance / "iter1_transcript.json"

    migrator.migrate_jobs_dir(jobs_dir)
    migrated_bytes = migrated_path.read_bytes()

    expected_path = tmp_path / "expected.json"
    expected_entries = CLAUDE.deserialize(legacy_raw)
    save_transcript(expected_path, expected_entries)
    assert migrated_bytes == expected_path.read_bytes()
    # And the round trip through TranscriptAdapter holds.
    assert TranscriptAdapter.validate_python(json.loads(migrated_bytes)) == expected_entries


def test_migrates_multi_file_job_dir(
    tmp_path: Path, migrator: ModuleType, legacy_raw: list[dict[str, Any]]
) -> None:
    jobs_dir = tmp_path / "jobs"
    provenance = _make_job(
        jobs_dir,
        "job-i",
        {
            "iter1_transcript.json": json.dumps(legacy_raw),
            "iter2_transcript.json": json.dumps(legacy_raw),
        },
    )

    counts = migrator.migrate_jobs_dir(jobs_dir)

    assert counts[migrator._MigrationOutcome.MIGRATED] == 2
    for name in ("iter1_transcript.json", "iter2_transcript.json"):
        target = provenance / name
        TranscriptAdapter.validate_json(target.read_bytes())
        assert target.with_suffix(target.suffix + ".legacy.json").is_file()


def test_migrates_mixed_shape_job_dir(
    tmp_path: Path, migrator: ModuleType, legacy_raw: list[dict[str, Any]]
) -> None:
    jobs_dir = tmp_path / "jobs"
    provenance = _make_job(
        jobs_dir,
        "job-j",
        {"iter1_transcript.json": json.dumps(legacy_raw)},
    )
    typed_target = provenance / "report_transcript.json"
    save_transcript(typed_target, [AssistantText(text="already typed")])
    typed_bytes_before = typed_target.read_bytes()

    counts = migrator.migrate_jobs_dir(jobs_dir)

    assert counts[migrator._MigrationOutcome.MIGRATED] == 1
    assert counts[migrator._MigrationOutcome.ALREADY_TYPED] == 1
    assert typed_target.read_bytes() == typed_bytes_before
    assert not typed_target.with_suffix(typed_target.suffix + ".legacy.json").exists()


def test_refuses_to_overwrite_existing_backup(
    tmp_path: Path, migrator: ModuleType, legacy_raw: list[dict[str, Any]]
) -> None:
    jobs_dir = tmp_path / "jobs"
    provenance = _make_job(
        jobs_dir,
        "job-k",
        {"iter1_transcript.json": json.dumps(legacy_raw)},
    )
    target = provenance / "iter1_transcript.json"
    backup = target.with_suffix(target.suffix + ".legacy.json")
    backup.write_bytes(b"prior backup sentinel")
    target_before = target.read_bytes()

    counts = migrator.migrate_jobs_dir(jobs_dir)

    assert counts[migrator._MigrationOutcome.BACKUP_EXISTS] == 1
    assert counts[migrator._MigrationOutcome.MIGRATED] == 0
    assert target.read_bytes() == target_before
    assert backup.read_bytes() == b"prior backup sentinel"


def test_unreadable_file_lands_in_unreadable_bucket(tmp_path: Path, migrator: ModuleType) -> None:
    jobs_dir = tmp_path / "jobs"
    provenance = jobs_dir / "job-l" / "provenance"
    provenance.mkdir(parents=True)
    target = provenance / "iter1_transcript.json"
    target.write_bytes(b"\xff\xfe not json")
    before = target.read_bytes()

    counts = migrator.migrate_jobs_dir(jobs_dir)

    assert counts[migrator._MigrationOutcome.UNREADABLE] == 1
    assert counts[migrator._MigrationOutcome.MIGRATED] == 0
    assert target.read_bytes() == before


def test_main_cli_smoke(
    tmp_path: Path,
    migrator: ModuleType,
    legacy_raw: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs_dir = tmp_path / "jobs"
    _make_job(jobs_dir, "job-m", {"iter1_transcript.json": json.dumps(legacy_raw)})

    monkeypatch.setattr(sys, "argv", ["migrate_legacy_transcripts.py", "--jobs-dir", str(jobs_dir)])
    rc = migrator.main()
    captured = capsys.readouterr()
    assert rc == 0
    assert "migrated: 1" in captured.out

    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr(sys, "argv", ["migrate_legacy_transcripts.py", "--jobs-dir", str(missing)])
    rc = migrator.main()
    assert rc == 2
