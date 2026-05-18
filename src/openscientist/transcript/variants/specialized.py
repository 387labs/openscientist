"""Specialised tool-event variants: shell execution, file change, web search,
collaborative agent calls."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ShellExecution(BaseModel):
    """A shell command (Codex ``CommandExecutionThreadItem``)."""

    type: Literal["shell_execution"] = "shell_execution"
    id: str
    command: str
    output: str
    exit_code: int | None = None
    command_actions: list[dict[str, Any]] | None = None
    status: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class FileChange(BaseModel):
    """A single file mutation.

    Codex ``FileChangeThreadItem.changes`` is split into one
    ``FileChange`` per ``FileUpdateChange`` so each entry is atomic;
    the shared parent ``id`` and ``status`` are carried on every entry.
    """

    type: Literal["file_change"] = "file_change"
    id: str
    path: str
    kind: Literal["write", "edit", "create", "delete", "rename"]
    diff: str | None = None
    success: bool
    status: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class WebSearch(BaseModel):
    """Codex ``WebSearchThreadItem``."""

    type: Literal["web_search"] = "web_search"
    id: str
    query: str
    action: dict[str, Any] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class CollabAgentToolCall(BaseModel):
    """Codex ``CollabAgentToolCallThreadItem`` (subagent spawn)."""

    type: Literal["collab_agent_tool_call"] = "collab_agent_tool_call"
    id: str
    prompt: str | None = None
    model: str | None = None
    agents_states: dict[str, dict[str, Any]] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
