"""Canonical file-IO helpers for persisting `list[TranscriptEntry]`."""

from __future__ import annotations

import os
from pathlib import Path

from openscientist.transcript.union import TranscriptAdapter, TranscriptEntry


def save_transcript(path: Path, entries: list[TranscriptEntry]) -> None:
    """Write ``entries`` to ``path`` as indented JSON, atomically.

    The serialised bytes are written to ``path`` with a ``.tmp``
    suffix appended and then renamed onto ``path`` via
    :func:`os.replace`, so a partial write never overwrites an
    existing target. Missing parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = TranscriptAdapter.dump_json(entries, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def load_transcript(path: Path) -> list[TranscriptEntry]:
    """Read ``path`` and validate it as ``list[TranscriptEntry]``.

    Raises :class:`FileNotFoundError` if ``path`` does not exist
    and :class:`pydantic.ValidationError` if its contents do not
    validate against the union.
    """
    return TranscriptAdapter.validate_json(path.read_bytes())
