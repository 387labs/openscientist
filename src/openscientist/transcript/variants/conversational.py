"""Conversational TranscriptEntry variants: user prompt, assistant text,
hidden reasoning."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class UserPrompt(BaseModel):
    """User-supplied prompt text.

    Distinct from :class:`~openscientist.transcript.variants.tool_io.ToolResult`,
    which is the SDK's user-role plumbing for returning tool output to the
    model. ``UserPrompt`` is for free-form prompts the orchestrator (or a
    human) feeds in.
    """

    type: Literal["user_prompt"] = "user_prompt"
    id: str | None = None
    text: str
    parent_tool_use_id: str | None = None
    uuid: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class AssistantText(BaseModel):
    """Model-produced visible text."""

    type: Literal["assistant_text"] = "assistant_text"
    id: str | None = None
    text: str
    model: str | None = None
    error: str | None = None
    parent_tool_use_id: str | None = None
    uuid: str | None = None
    phase: str | None = None
    memory_citation: dict[str, Any] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class Reasoning(BaseModel):
    """Hidden chain-of-thought tokens.

    Populated by Claude ``ThinkingBlock`` (with ``signature``) and by
    Codex ``ReasoningThreadItem`` (with ``content`` + ``summary`` lists).
    """

    type: Literal["reasoning"] = "reasoning"
    id: str | None = None
    text: str
    summary: str | list[str] | None = None
    signature: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
