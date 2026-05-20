"""Agent marker types and the :class:`TranscriptDeserializer` Protocol."""

from typing import Any, Protocol, runtime_checkable

from openscientist.transcript.union import TranscriptEntry


class AgentMarker:
    """Marker base for backend agents. Pure tag, never instantiated."""


class ClaudeAgent(AgentMarker):
    """Marker for the ``claude-agent-sdk`` wire format."""


class CodexAgent(AgentMarker):
    """Marker for the ``openai-codex`` wire formats."""


@runtime_checkable
class TranscriptDeserializer[A: AgentMarker](Protocol):
    """Translates a backend's native wire transcript into a
    :data:`TranscriptEntry` list.
    """

    def deserialize(self, raw: list[dict[str, Any]]) -> list[TranscriptEntry]:
        """Return the typed entries equivalent to ``raw``.

        Implementations MUST preserve every source key (typed field
        or ``raw`` overlay) and MUST emit :class:`UnknownEntry` with
        a logged ``WARNING`` for unrecognised shapes.
        """
