"""Normalized transcript schema for agent iteration outputs.

The :data:`TranscriptEntry` discriminated union represents every
backend's transcript in one shape. Backends will ship
:class:`TranscriptDeserializer` implementations that map their
native message shape into it without dropping any source field;
unrecognised shapes become :class:`UnknownEntry`.
"""

from openscientist.transcript.agents import (
    AgentMarker,
    ClaudeAgent,
    CodexAgent,
    TranscriptDeserializer,
)
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
    "AgentMarker",
    "AssistantText",
    "ClaudeAgent",
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
