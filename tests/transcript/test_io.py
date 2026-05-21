"""Tests for the transcript file-IO helpers."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from openscientist.transcript import (
    CLAUDE,
    AssistantText,
    ToolCall,
    ToolResult,
    TranscriptAdapter,
    TranscriptEntry,
    load_transcript,
    save_transcript,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "transcripts" / "claude_agent_sdk"


@pytest.fixture
def sample_entries() -> list[TranscriptEntry]:
    return [
        AssistantText(text="hello"),
        ToolCall(id="c1", tool="execute_code", arguments={"code": "print(1)"}),
        ToolResult(call_id="c1", output="1\n", success=True),
    ]


def test_roundtrip_non_empty(tmp_path: Path, sample_entries: list[TranscriptEntry]) -> None:
    target = tmp_path / "iter01_transcript.json"
    save_transcript(target, sample_entries)
    loaded = load_transcript(target)
    assert loaded == sample_entries


def test_roundtrip_empty(tmp_path: Path) -> None:
    target = tmp_path / "empty.json"
    save_transcript(target, [])
    assert load_transcript(target) == []


def test_save_creates_missing_parent_dirs(
    tmp_path: Path, sample_entries: list[TranscriptEntry]
) -> None:
    target = tmp_path / "deeply" / "nested" / "iter01_transcript.json"
    assert not target.parent.exists()
    save_transcript(target, sample_entries)
    assert target.is_file()
    assert load_transcript(target) == sample_entries


def test_save_is_atomic_and_leaves_no_tmp_on_success(
    tmp_path: Path, sample_entries: list[TranscriptEntry]
) -> None:
    target = tmp_path / "iter01_transcript.json"
    save_transcript(target, sample_entries)
    siblings = list(tmp_path.iterdir())
    assert siblings == [target], f"unexpected sibling files left behind: {siblings}"


def test_save_failure_does_not_clobber_existing_target(
    tmp_path: Path, sample_entries: list[TranscriptEntry]
) -> None:
    target = tmp_path / "iter01_transcript.json"
    save_transcript(target, sample_entries)
    original_bytes = target.read_bytes()
    # Force the serialiser to raise. The target must stay intact and
    # no .tmp sibling must be left behind.
    with patch.object(TranscriptAdapter, "dump_json", side_effect=RuntimeError("forced failure")):
        with pytest.raises(RuntimeError, match="forced failure"):
            save_transcript(target, [AssistantText(text="should not land")])
    assert target.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [target]


def test_save_failure_removes_partial_tmp_when_target_did_not_exist(
    tmp_path: Path,
) -> None:
    target = tmp_path / "iter01_transcript.json"
    # Patch os.replace so the bytes land on the .tmp path but the
    # rename step blows up. The .tmp must be cleaned up.
    import openscientist.transcript.io as transcript_io

    with patch.object(transcript_io.os, "replace", side_effect=OSError("rename failed")):
        with pytest.raises(OSError, match="rename failed"):
            save_transcript(target, [AssistantText(text="x")])
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_load_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_transcript(tmp_path / "does_not_exist.json")


def test_load_unknown_discriminator_raises_validation_error(tmp_path: Path) -> None:
    target = tmp_path / "bad.json"
    target.write_text(json.dumps([{"type": "definitely_not_a_real_variant"}]))
    with pytest.raises(ValidationError):
        load_transcript(target)


def test_save_output_is_indented(tmp_path: Path, sample_entries: list[TranscriptEntry]) -> None:
    target = tmp_path / "iter01_transcript.json"
    save_transcript(target, sample_entries)
    text = target.read_text()
    assert "\n  " in text, "expected pretty-printed JSON (indent=2)"


def test_roundtrip_real_claude_fixture_through_deserializer(tmp_path: Path) -> None:
    """`save_transcript` + `load_transcript` compose cleanly with the
    `CLAUDE` deserializer on a real captured Claude transcript."""
    source = FIXTURE_DIR / "capital_of_france_iter04.json.gz"
    with gzip.open(source, "rt", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)
    entries = CLAUDE.deserialize(raw)
    assert entries

    target = tmp_path / "iter04.json"
    save_transcript(target, entries)
    loaded = load_transcript(target)
    assert loaded == entries
