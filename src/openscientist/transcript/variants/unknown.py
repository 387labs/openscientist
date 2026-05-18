"""The last-resort ``UnknownEntry`` TranscriptEntry variant."""

from typing import Any, Literal

from pydantic import BaseModel


class UnknownEntry(BaseModel):
    """A source entry the translator does not recognise.

    Recorded rather than dropped, so the no-drop invariant holds even
    for SDK shapes that postdate this code. Each translator that emits
    one MUST also log a WARNING naming the source SDK and the
    unrecognised ``type`` discriminator so live drift surfaces in
    operational logs even before tests catch it.

    Tests count ``UnknownEntry`` occurrences against committed
    fixtures: a nonzero count in a real-data fixture fails CI and
    forces a translator update.
    """

    type: Literal["unknown_entry"] = "unknown_entry"
    source: Literal["claude", "codex"]
    raw: dict[str, Any]
