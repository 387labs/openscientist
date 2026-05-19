"""Normalized transcript schema for agent iteration outputs.

The :data:`TranscriptEntry` discriminated union represents every
backend's transcript in one shape. Per-backend
:class:`TranscriptDeserializer` implementations map native message
shapes into it without dropping any source field. Unrecognised
shapes become :class:`UnknownEntry`.

See :data:`CLAUDE` for the canonical Claude deserializer.
"""

from openscientist.transcript.agents import (
    AgentMarker,
    ClaudeAgent,
    CodexAgent,
    TranscriptDeserializer,
)
from openscientist.transcript.translators import CLAUDE, ClaudeDeserializer
from openscientist.transcript.union import TranscriptAdapter, TranscriptEntry
from openscientist.transcript.variants import (
    AssistantText,
    CollabAgentToolCall,
    FileChange,
    HookPrompt,
    ImageGeneration,
    ImageView,
    Plan,
    Reasoning,
    ReviewModeEntered,
    ReviewModeExited,
    SessionInit,
    ShellExecution,
    TaskNotification,
    TaskProgress,
    TaskStarted,
    ToolCall,
    ToolResult,
    UnknownEntry,
    UserPrompt,
    WebSearch,
)

__all__ = [
    "CLAUDE",
    "AgentMarker",
    "AssistantText",
    "ClaudeAgent",
    "ClaudeDeserializer",
    "CodexAgent",
    "CollabAgentToolCall",
    "FileChange",
    "HookPrompt",
    "ImageGeneration",
    "ImageView",
    "Plan",
    "Reasoning",
    "ReviewModeEntered",
    "ReviewModeExited",
    "SessionInit",
    "ShellExecution",
    "TaskNotification",
    "TaskProgress",
    "TaskStarted",
    "ToolCall",
    "ToolResult",
    "TranscriptAdapter",
    "TranscriptDeserializer",
    "TranscriptEntry",
    "UnknownEntry",
    "UserPrompt",
    "WebSearch",
]
