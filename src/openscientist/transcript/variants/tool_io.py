"""Generic tool-call/tool-result TranscriptEntry variants."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool invocation requested by the agent.

    The ``id`` is the backend-assigned call identifier and is referenced
    by the paired :class:`ToolResult` via ``call_id``.
    """

    type: Literal["tool_call"] = "tool_call"
    id: str
    tool: str
    arguments: dict[str, Any]
    server: str | None = None
    namespace: str | None = None
    parent_tool_use_id: str | None = None
    uuid: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of a :class:`ToolCall`, paired by ``call_id``."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    output: str
    success: bool
    status: str | None = None
    duration_ms: int | None = None
    structured_content: Any | None = None
    content_items: list[Any] | None = None
    tool_use_result: dict[str, Any] | None = None
    error_message: str | None = None
    mcp_app_resource_uri: str | None = None
    parent_tool_use_id: str | None = None
    uuid: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
