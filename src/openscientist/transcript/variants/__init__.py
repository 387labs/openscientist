"""TranscriptEntry variants grouped by semantic family."""

from openscientist.transcript.variants.conversational import (
    AssistantText,
    Reasoning,
    UserPrompt,
)
from openscientist.transcript.variants.planning import HookPrompt, Plan
from openscientist.transcript.variants.review_mode import (
    ReviewModeEntered,
    ReviewModeExited,
)
from openscientist.transcript.variants.specialized import (
    CollabAgentToolCall,
    FileChange,
    ShellExecution,
    WebSearch,
)
from openscientist.transcript.variants.task_lifecycle import (
    SessionInit,
    TaskNotification,
    TaskProgress,
    TaskStarted,
)
from openscientist.transcript.variants.tool_io import ToolCall, ToolResult
from openscientist.transcript.variants.unknown import UnknownEntry
from openscientist.transcript.variants.visual import ImageGeneration, ImageView

__all__ = [
    "AssistantText",
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
    "UnknownEntry",
    "UserPrompt",
    "WebSearch",
]
