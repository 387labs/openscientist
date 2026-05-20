"""Claude SystemMessage session/task-lifecycle TranscriptEntry variants:
SessionInit, TaskStarted, TaskProgress, TaskNotification."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SessionInit(BaseModel):
    """Claude session bootstrap (``SystemMessage`` ``subtype="init"``).

    Carries the session config snapshot: tools, slash commands,
    subagents, model, cwd.
    """

    type: Literal["session_init"] = "session_init"
    session_id: str | None = None
    uuid: str | None = None
    cwd: str | None = None
    model: str | None = None
    permission_mode: str | None = None
    api_key_source: str | None = None
    tools: list[str] | None = None
    slash_commands: list[str] | None = None
    agents: list[str] | None = None
    mcp_servers: list[Any] | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TaskStarted(BaseModel):
    """Claude ``TaskStartedMessage``."""

    type: Literal["task_started"] = "task_started"
    task_id: str
    description: str
    task_type: str | None = None
    parent_tool_use_id: str | None = None
    session_id: str | None = None
    uuid: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TaskProgress(BaseModel):
    """Claude ``TaskProgressMessage``."""

    type: Literal["task_progress"] = "task_progress"
    task_id: str
    description: str
    last_tool_name: str | None = None
    usage: dict[str, Any] | None = None
    parent_tool_use_id: str | None = None
    session_id: str | None = None
    uuid: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TaskNotification(BaseModel):
    """Claude ``TaskNotificationMessage``."""

    type: Literal["task_notification"] = "task_notification"
    task_id: str
    status: str
    summary: str
    output_file: str
    usage: dict[str, Any] | None = None
    parent_tool_use_id: str | None = None
    session_id: str | None = None
    uuid: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
