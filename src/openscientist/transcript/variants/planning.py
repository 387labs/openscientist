"""Planning and hook-prompt TranscriptEntry variants."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Codex ``PlanThreadItem`` (agent-emitted plan text)."""

    type: Literal["plan"] = "plan"
    id: str
    text: str
    raw: dict[str, Any] = Field(default_factory=dict)


class HookPrompt(BaseModel):
    """Codex ``HookPromptThreadItem`` (config-injected prompt fragments)."""

    type: Literal["hook_prompt"] = "hook_prompt"
    id: str
    fragments: list[dict[str, Any]]
    raw: dict[str, Any] = Field(default_factory=dict)
