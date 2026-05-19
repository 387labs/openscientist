"""Claude (``claude-agent-sdk``) :class:`TranscriptDeserializer` backend."""

from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from openscientist.transcript.translators.helpers import (
    block_extras_or_none,
    coerce_tool_result_content,
    full_block_overlay,
    merge_overlay,
    safe_str,
    unknown,
    unknown_block,
    wrapper_overlay,
)
from openscientist.transcript.union import TranscriptEntry
from openscientist.transcript.variants import (
    AssistantText,
    Reasoning,
    SessionInit,
    TaskNotification,
    TaskProgress,
    TaskStarted,
    ToolCall,
    ToolResult,
    UserPrompt,
)

# ---------------------------------------------------------------------------
# Wrapper-level role contexts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _AssistantContext:
    """Wrapper metadata lifted from the assistant message."""

    model: str | None
    error: str | None
    parent_tool_use_id: str | None
    uuid: str | None
    overlay: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _UserContext:
    """Wrapper metadata for user-role entries."""

    parent_tool_use_id: str | None
    uuid: str | None
    tool_use_result: dict[str, Any] | None
    overlay: dict[str, Any]


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class _ClaudeBlock(BaseModel):
    """Base for Claude content blocks."""

    model_config = ConfigDict(extra="allow")

    def model_extras(self) -> dict[str, Any]:
        return dict(self.__pydantic_extra__ or {})


class _TextBlock(_ClaudeBlock):
    type: Literal["text"]
    text: str

    def translate(self, ctx: _AssistantContext | _UserContext) -> TranscriptEntry:
        raw = merge_overlay(ctx.overlay, block_extras_or_none(self.model_extras()))
        if isinstance(ctx, _AssistantContext):
            return AssistantText(
                text=self.text,
                model=ctx.model,
                error=ctx.error,
                parent_tool_use_id=ctx.parent_tool_use_id,
                uuid=ctx.uuid,
                raw=raw,
            )
        return UserPrompt(
            text=self.text,
            parent_tool_use_id=ctx.parent_tool_use_id,
            uuid=ctx.uuid,
            raw=raw,
        )


class _ToolUseBlock(_ClaudeBlock):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)

    def translate(self, ctx: _AssistantContext) -> TranscriptEntry:
        return ToolCall(
            id=self.id,
            tool=self.name,
            arguments=dict(self.input),
            parent_tool_use_id=ctx.parent_tool_use_id,
            uuid=ctx.uuid,
            raw=merge_overlay(ctx.overlay, block_extras_or_none(self.model_extras())),
        )


class _ToolResultBlock(_ClaudeBlock):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Any | None = None
    is_error: bool | None = None

    def translate(self, ctx: _UserContext) -> TranscriptEntry:
        # Preserve the source content list. ``coerce_tool_result_content``
        # only joins text blocks, so non-text entries (e.g. tool_reference
        # tags) would otherwise be lost.
        content_items = (
            [dict(item) if isinstance(item, dict) else item for item in self.content]
            if isinstance(self.content, list)
            else None
        )
        return ToolResult(
            call_id=self.tool_use_id,
            output=coerce_tool_result_content(self.content),
            success=not bool(self.is_error),
            content_items=content_items,
            tool_use_result=ctx.tool_use_result,
            parent_tool_use_id=ctx.parent_tool_use_id,
            uuid=ctx.uuid,
            raw=merge_overlay(ctx.overlay, block_extras_or_none(self.model_extras())),
        )


class _ThinkingBlock(_ClaudeBlock):
    type: Literal["thinking"]
    thinking: str | None = None
    text: str | None = None
    summary: str | list[str] | None = None
    signature: str | None = None

    def translate(self, ctx: _AssistantContext) -> TranscriptEntry:
        text = self.thinking if isinstance(self.thinking, str) else self.text
        if not isinstance(text, str):
            return unknown_block(
                "claude", ctx.overlay, self, "thinking block missing thinking text"
            )
        return Reasoning(
            text=text,
            summary=self.summary,
            signature=self.signature,
            raw=merge_overlay(ctx.overlay, block_extras_or_none(self.model_extras())),
        )


_AssistantBlock = Annotated[
    _TextBlock | _ToolUseBlock | _ThinkingBlock,
    Field(discriminator="type"),
]
_UserBlock = Annotated[
    _ToolResultBlock | _TextBlock,
    Field(discriminator="type"),
]

_AssistantBlockAdapter: TypeAdapter[_AssistantBlock] = TypeAdapter(_AssistantBlock)
_UserBlockAdapter: TypeAdapter[_UserBlock] = TypeAdapter(_UserBlock)


# ---------------------------------------------------------------------------
# Wrapper-level key sets (so we know what is "consumed" vs. "leftover")
# ---------------------------------------------------------------------------

# Keys consumed by typed fields per entry shape. Anything else lands
# in the produced entry's ``raw`` dict.
_ASSISTANT_ENTRY_CONSUMED: frozenset[str] = frozenset(
    {"type", "message", "parent_tool_use_id", "uuid"}
)
_USER_ENTRY_CONSUMED: frozenset[str] = frozenset(
    {"type", "message", "parent_tool_use_id", "uuid", "tool_use_result"}
)
_ASSISTANT_MESSAGE_CONSUMED: frozenset[str] = frozenset({"content", "model", "error"})
_USER_MESSAGE_CONSUMED: frozenset[str] = frozenset({"content"})
_SYSTEM_TASK_STARTED_CONSUMED: frozenset[str] = frozenset(
    {"type", "subtype", "task_id", "description", "task_type", "tool_use_id", "session_id", "uuid"}
)
_SYSTEM_TASK_PROGRESS_CONSUMED: frozenset[str] = frozenset(
    {
        "type",
        "subtype",
        "task_id",
        "description",
        "last_tool_name",
        "usage",
        "tool_use_id",
        "session_id",
        "uuid",
    }
)
_SYSTEM_TASK_NOTIFICATION_CONSUMED: frozenset[str] = frozenset(
    {
        "type",
        "subtype",
        "task_id",
        "status",
        "summary",
        "output_file",
        "usage",
        "tool_use_id",
        "session_id",
        "uuid",
    }
)
_SYSTEM_INIT_CONSUMED: frozenset[str] = frozenset(
    {
        "type",
        "subtype",
        "session_id",
        "uuid",
        "cwd",
        "model",
        "permissionMode",
        "apiKeySource",
        "tools",
        "slash_commands",
        "agents",
        "mcp_servers",
    }
)


def _leftover(d: dict[str, Any], consumed: frozenset[str]) -> dict[str, Any]:
    """Return ``{k: v for k, v in d.items() if k not in consumed}``."""
    return {k: v for k, v in d.items() if k not in consumed}


# ---------------------------------------------------------------------------
# ClaudeDeserializer
# ---------------------------------------------------------------------------


class ClaudeDeserializer:
    """:class:`TranscriptDeserializer` for the ``claude-agent-sdk``
    wire format. Stateless. See :data:`CLAUDE` for the canonical
    instance.
    """

    def deserialize(self, raw: list[dict[str, Any]]) -> list[TranscriptEntry]:
        """Translate ``raw`` into typed :class:`TranscriptEntry` instances."""
        out: list[TranscriptEntry] = []
        for entry in raw:
            if not isinstance(entry, dict):
                out.append(
                    unknown(
                        "claude",
                        {"_non_dict_entry": repr(entry)},
                        "non-dict source entry",
                    )
                )
                continue
            out.extend(self._translate_entry(entry))
        return out

    # ---- per-role dispatch ----

    def _translate_entry(self, entry: dict[str, Any]) -> list[TranscriptEntry]:
        match entry.get("type"):
            case "assistant":
                return self._translate_assistant(entry)
            case "user":
                return self._translate_user(entry)
            case "system":
                return self._translate_system(entry)
            case other:
                return [
                    unknown(
                        "claude",
                        dict(entry),
                        f"unrecognised entry type {other!r}",
                    )
                ]

    def _translate_assistant(self, entry: dict[str, Any]) -> list[TranscriptEntry]:
        message = entry.get("message")
        if not isinstance(message, dict):
            return [
                unknown(
                    "claude",
                    dict(entry),
                    "assistant entry missing dict-shaped message",
                )
            ]
        overlay = wrapper_overlay(
            _leftover(entry, _ASSISTANT_ENTRY_CONSUMED),
            _leftover(message, _ASSISTANT_MESSAGE_CONSUMED),
        )
        ctx = _AssistantContext(
            model=message.get("model"),
            error=message.get("error"),
            parent_tool_use_id=entry.get("parent_tool_use_id"),
            uuid=entry.get("uuid"),
            overlay=overlay,
        )
        content = message.get("content", [])
        if not isinstance(content, list):
            return [
                unknown(
                    "claude",
                    dict(entry),
                    "assistant message.content is not a list",
                )
            ]
        return [self._translate_assistant_block(b, ctx) for b in content]

    def _translate_user(self, entry: dict[str, Any]) -> list[TranscriptEntry]:
        message = entry.get("message")
        if not isinstance(message, dict):
            return [
                unknown(
                    "claude",
                    dict(entry),
                    "user entry missing dict-shaped message",
                )
            ]
        overlay = wrapper_overlay(
            _leftover(entry, _USER_ENTRY_CONSUMED),
            _leftover(message, _USER_MESSAGE_CONSUMED),
        )
        ctx = _UserContext(
            parent_tool_use_id=entry.get("parent_tool_use_id"),
            uuid=entry.get("uuid"),
            tool_use_result=entry.get("tool_use_result"),
            overlay=overlay,
        )
        content = message.get("content", [])
        if isinstance(content, str):
            return [
                UserPrompt(
                    text=content,
                    parent_tool_use_id=ctx.parent_tool_use_id,
                    uuid=ctx.uuid,
                    raw=overlay,
                )
            ]
        if not isinstance(content, list):
            return [
                unknown(
                    "claude",
                    dict(entry),
                    "user message.content is neither str nor list",
                )
            ]
        return [self._translate_user_block(b, ctx) for b in content]

    def _translate_system(self, entry: dict[str, Any]) -> list[TranscriptEntry]:
        subtype = entry.get("subtype")
        match subtype:
            case "task_started":
                return [
                    TaskStarted(
                        task_id=safe_str(entry.get("task_id"), "task_started.task_id"),
                        description=safe_str(entry.get("description"), "task_started.description"),
                        task_type=entry.get("task_type"),
                        parent_tool_use_id=entry.get("tool_use_id"),
                        session_id=entry.get("session_id"),
                        uuid=entry.get("uuid"),
                        raw=_leftover(entry, _SYSTEM_TASK_STARTED_CONSUMED),
                    )
                ]
            case "task_progress":
                usage = entry.get("usage")
                return [
                    TaskProgress(
                        task_id=safe_str(entry.get("task_id"), "task_progress.task_id"),
                        description=safe_str(entry.get("description"), "task_progress.description"),
                        last_tool_name=entry.get("last_tool_name"),
                        usage=usage if isinstance(usage, dict) else None,
                        parent_tool_use_id=entry.get("tool_use_id"),
                        session_id=entry.get("session_id"),
                        uuid=entry.get("uuid"),
                        raw=_leftover(entry, _SYSTEM_TASK_PROGRESS_CONSUMED),
                    )
                ]
            case "task_notification":
                usage = entry.get("usage")
                return [
                    TaskNotification(
                        task_id=safe_str(entry.get("task_id"), "task_notification.task_id"),
                        status=safe_str(entry.get("status"), "task_notification.status"),
                        summary=safe_str(entry.get("summary"), "task_notification.summary"),
                        output_file=safe_str(
                            entry.get("output_file"), "task_notification.output_file"
                        ),
                        usage=usage if isinstance(usage, dict) else None,
                        parent_tool_use_id=entry.get("tool_use_id"),
                        session_id=entry.get("session_id"),
                        uuid=entry.get("uuid"),
                        raw=_leftover(entry, _SYSTEM_TASK_NOTIFICATION_CONSUMED),
                    )
                ]
            case "init":
                tools = entry.get("tools")
                slash_commands = entry.get("slash_commands")
                agents = entry.get("agents")
                mcp_servers = entry.get("mcp_servers")
                return [
                    SessionInit(
                        session_id=entry.get("session_id"),
                        uuid=entry.get("uuid"),
                        cwd=entry.get("cwd"),
                        model=entry.get("model"),
                        permission_mode=entry.get("permissionMode"),
                        api_key_source=entry.get("apiKeySource"),
                        tools=tools if isinstance(tools, list) else None,
                        slash_commands=(
                            slash_commands if isinstance(slash_commands, list) else None
                        ),
                        agents=agents if isinstance(agents, list) else None,
                        mcp_servers=(mcp_servers if isinstance(mcp_servers, list) else None),
                        raw=_leftover(entry, _SYSTEM_INIT_CONSUMED),
                    )
                ]
            case other:
                return [
                    unknown(
                        "claude",
                        dict(entry),
                        f"unrecognised system subtype {other!r}",
                    )
                ]

    # ---- per-block dispatch ----

    def _translate_assistant_block(
        self, raw_block: object, ctx: _AssistantContext
    ) -> TranscriptEntry:
        if not isinstance(raw_block, dict):
            return unknown(
                "claude",
                {"_non_dict_block": repr(raw_block), "_overlay": ctx.overlay},
                "assistant content block is not a dict",
            )
        try:
            block = _AssistantBlockAdapter.validate_python(raw_block)
        except ValidationError as exc:
            return unknown(
                "claude",
                full_block_overlay(raw_block, ctx.overlay),
                f"unrecognised assistant block type {raw_block.get('type')!r}: "
                f"{exc.errors()[0]['msg']}",
            )
        return block.translate(ctx)

    def _translate_user_block(self, raw_block: object, ctx: _UserContext) -> TranscriptEntry:
        if not isinstance(raw_block, dict):
            return unknown(
                "claude",
                {"_non_dict_block": repr(raw_block), "_overlay": ctx.overlay},
                "user content block is not a dict",
            )
        try:
            block = _UserBlockAdapter.validate_python(raw_block)
        except ValidationError as exc:
            return unknown(
                "claude",
                full_block_overlay(raw_block, ctx.overlay),
                f"unrecognised user block type {raw_block.get('type')!r}: {exc.errors()[0]['msg']}",
            )
        return block.translate(ctx)


CLAUDE: ClaudeDeserializer = ClaudeDeserializer()
"""Canonical :class:`TranscriptDeserializer` for the Claude wire format."""
