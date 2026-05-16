"""Normalized transcript schema for agent iteration outputs.

Every agent backend (claude-agent-sdk, codex-sdk, future backends) emits its
own native message format. The orchestrator and the webapp need a single
shape they can parse independently of which backend produced the run.

This module defines that shape as a Pydantic discriminated union: each
variant is a typed class with a ``type`` literal discriminator. New
variants can be added by appending to the ``TranscriptEntry`` union.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class AssistantText(BaseModel):
    """A piece of natural-language text produced by the agent."""

    type: Literal["assistant_text"] = "assistant_text"
    text: str


class ToolCall(BaseModel):
    """A tool invocation requested by the agent.

    The ``id`` is the backend-assigned call identifier and is referenced by
    the paired :class:`ToolResult` via ``call_id``.
    """

    type: Literal["tool_call"] = "tool_call"
    id: str
    tool: str
    arguments: dict[str, object]


class ToolResult(BaseModel):
    """The outcome of a :class:`ToolCall`, paired by ``call_id``.

    ``output`` is the serialized text the agent observes. ``success`` is
    ``True`` when the tool ran to completion without raising. ``duration_ms``
    is the wall-clock duration when the backend reports it.
    """

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    output: str
    success: bool
    duration_ms: int | None = None


class ShellExecution(BaseModel):
    """A shell command the agent ran via its backend's built-in shell tool.

    Carries the merged stdout/stderr in ``output`` and the process
    ``exit_code`` when the backend reports it.
    """

    type: Literal["shell_execution"] = "shell_execution"
    id: str
    command: str
    output: str
    exit_code: int | None = None


class FileChange(BaseModel):
    """A single file mutation made by the agent.

    The ``kind`` distinguishes whole-file writes from in-place edits. The
    ``diff`` is the textual representation of the change when the backend
    provides one (always for Codex, sometimes for claude-agent-sdk).
    """

    type: Literal["file_change"] = "file_change"
    id: str
    path: str
    kind: Literal["write", "edit"]
    diff: str | None = None
    success: bool


class Reasoning(BaseModel):
    """Hidden chain-of-thought tokens produced by reasoning-capable models.

    Populated by OpenAI o-series reasoning blocks and by Anthropic extended-
    thinking content when the SDK exposes them as a distinct stream.
    """

    type: Literal["reasoning"] = "reasoning"
    text: str
    summary: str | None = None


TranscriptEntry = Annotated[
    AssistantText | ToolCall | ToolResult | ShellExecution | FileChange | Reasoning,
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
