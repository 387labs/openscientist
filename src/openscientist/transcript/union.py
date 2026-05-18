"""The :data:`TranscriptEntry` discriminated union and its
:data:`TranscriptAdapter` :class:`~pydantic.TypeAdapter`.

Each variant class defines a ``type: Literal[...]`` discriminator. The
union dispatches by ``type`` at validation time, so a JSON list of
heterogeneous transcript entries round-trips through
``TranscriptAdapter.validate_python`` into the correct typed variants.
"""

from typing import Annotated

from pydantic import Field, TypeAdapter

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

TranscriptEntry = Annotated[
    UserPrompt
    | AssistantText
    | Reasoning
    | ToolCall
    | ToolResult
    | ShellExecution
    | FileChange
    | WebSearch
    | CollabAgentToolCall
    | ImageView
    | ImageGeneration
    | Plan
    | HookPrompt
    | SessionInit
    | TaskStarted
    | TaskProgress
    | TaskNotification
    | ReviewModeEntered
    | ReviewModeExited
    | UnknownEntry,
    Field(discriminator="type"),
]
"""Discriminated union of every transcript variant.

Use with :class:`pydantic.TypeAdapter` for round-trip parsing and
serialization of lists of entries::

    adapter = TypeAdapter(list[TranscriptEntry])
    entries = adapter.validate_python(loaded_json_list)
    raw = adapter.dump_python(entries, mode="json")
"""


TranscriptAdapter: TypeAdapter[list[TranscriptEntry]] = TypeAdapter(list[TranscriptEntry])
"""Convenience adapter for the common ``list[TranscriptEntry]`` shape.

Holds the compiled pydantic-core validator and serializer so each round
trip avoids the construction cost.
"""
