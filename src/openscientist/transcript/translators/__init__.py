"""Backend-specific :class:`~openscientist.transcript.agents.TranscriptDeserializer`
implementations producing :data:`~openscientist.transcript.union.TranscriptEntry`
lists."""

from openscientist.transcript.translators.claude import CLAUDE, ClaudeDeserializer
from openscientist.transcript.translators.codex import CODEX, CodexDeserializer

__all__ = ["CLAUDE", "CODEX", "ClaudeDeserializer", "CodexDeserializer"]
