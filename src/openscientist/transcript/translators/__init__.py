"""Backend-specific :class:`~openscientist.transcript.agents.TranscriptDeserializer`
implementations producing :data:`~openscientist.transcript.union.TranscriptEntry`
lists."""

from openscientist.transcript.translators.claude import CLAUDE, ClaudeDeserializer

__all__ = ["CLAUDE", "ClaudeDeserializer"]
