"""Codex CLI exec :class:`TranscriptDeserializer` backend.

Translates the ``ThreadEvent`` JSONL stream emitted by
``codex exec --json``. Wire format declared in
``codex-rs/exec/src/exec_events.rs``.
"""

from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from openscientist.transcript.translators.helpers import (
    block_extras_or_none,
    merge_overlay,
    unknown,
    unknown_block,
)
from openscientist.transcript.union import TranscriptEntry
from openscientist.transcript.variants import (
    AssistantText,
    CollabAgentToolCall,
    FileChange,
    Plan,
    Reasoning,
    ShellExecution,
    TaskNotification,
    TaskStarted,
    ToolCall,
    ToolResult,
    UserPrompt,
    WebSearch,
)

# ---------------------------------------------------------------------------
# Item-level context (passed to each item's translate method)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CodexItemContext:
    """Stream metadata threaded into each item's translate call."""

    thread_id: str | None
    overlay: dict[str, Any]


# ---------------------------------------------------------------------------
# Item-level Pydantic models (discriminated by ``type``)
# ---------------------------------------------------------------------------


class _CodexItem(BaseModel):
    """Base for Codex CLI exec items."""

    model_config = ConfigDict(extra="allow")
    id: str

    def model_extras(self) -> dict[str, Any]:
        return dict(self.__pydantic_extra__ or {})

    def _raw(self, ctx: _CodexItemContext) -> dict[str, Any]:
        return merge_overlay(ctx.overlay, block_extras_or_none(self.model_extras()))


class _AgentMessageItem(_CodexItem):
    type: Literal["agent_message"]
    text: str

    def translate(self, ctx: _CodexItemContext) -> list[TranscriptEntry]:
        return [AssistantText(id=self.id, text=self.text, raw=self._raw(ctx))]


class _ReasoningItem(_CodexItem):
    type: Literal["reasoning"]
    text: str

    def translate(self, ctx: _CodexItemContext) -> list[TranscriptEntry]:
        return [Reasoning(id=self.id, text=self.text, raw=self._raw(ctx))]


class _CommandExecutionItem(_CodexItem):
    type: Literal["command_execution"]
    command: str
    aggregated_output: str | None = None
    exit_code: int | None = None
    status: str | None = None

    def translate(self, ctx: _CodexItemContext) -> list[TranscriptEntry]:
        return [
            ShellExecution(
                id=self.id,
                command=self.command,
                output=self.aggregated_output or "",
                exit_code=self.exit_code,
                status=self.status,
                raw=self._raw(ctx),
            )
        ]


class _FileUpdateChange(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str
    kind: str  # PatchChangeKind: "add" | "delete" | "update"


_FILE_CHANGE_KIND_TO_VARIANT: dict[str, str] = {
    "add": "create",
    "delete": "delete",
    "update": "edit",
}


class _FileChangeItem(_CodexItem):
    type: Literal["file_change"]
    changes: list[_FileUpdateChange] = Field(default_factory=list)
    status: str | None = None  # PatchApplyStatus

    def translate(self, ctx: _CodexItemContext) -> list[TranscriptEntry]:
        if not self.changes:
            return [unknown_block("codex", ctx.overlay, self, "file_change item has no changes")]
        success = self.status == "completed"
        out: list[TranscriptEntry] = []
        # One FileChange entry per source change, sharing parent id + status.
        for change in self.changes:
            kind = _FILE_CHANGE_KIND_TO_VARIANT.get(change.kind)
            if kind is None:
                out.append(
                    unknown(
                        "codex",
                        {
                            "_block": {
                                **self.model_dump(mode="json"),
                                "_offending_change": change.model_dump(mode="json"),
                            },
                            "_overlay": ctx.overlay,
                        },
                        f"file_change kind {change.kind!r} has no FileChange variant",
                    )
                )
                continue
            out.append(
                FileChange(
                    id=self.id,
                    path=change.path,
                    kind=kind,  # type: ignore[arg-type]  # validated above
                    success=success,
                    status=self.status,
                    raw=self._raw(ctx),
                )
            )
        return out


class _McpToolCallItemResult(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    content: list[Any] = Field(default_factory=list)
    structured_content: Any | None = None
    meta: Any | None = Field(default=None, alias="_meta")


class _McpToolCallItemError(BaseModel):
    model_config = ConfigDict(extra="allow")
    message: str


class _McpToolCallItem(_CodexItem):
    type: Literal["mcp_tool_call"]
    server: str
    tool: str
    arguments: Any = None
    result: _McpToolCallItemResult | None = None
    error: _McpToolCallItemError | None = None
    status: str | None = None

    def translate(self, ctx: _CodexItemContext) -> list[TranscriptEntry]:
        # Emit ToolCall + ToolResult paired on id.
        args = self.arguments if isinstance(self.arguments, dict) else {}
        raw = self._raw(ctx)
        if self.arguments is not None and not isinstance(self.arguments, dict):
            raw = dict(raw)
            raw["_arguments_non_dict"] = self.arguments
        out: list[TranscriptEntry] = [
            ToolCall(
                id=self.id,
                tool=self.tool,
                arguments=args,
                server=self.server,
                raw=raw,
            )
        ]
        success = self.error is None and self.status == "completed"
        output_text = ""
        structured: Any | None = None
        content_items: list[Any] | None = None
        if self.result is not None:
            structured = self.result.structured_content
            # Preserve the source content list. Non-text blocks would
            # otherwise be lost by the text-only flatten below.
            content_items = [
                dict(item) if isinstance(item, dict) else item for item in self.result.content
            ]
            parts: list[str] = []
            for item in self.result.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            output_text = "\n".join(parts)
        out.append(
            ToolResult(
                call_id=self.id,
                output=output_text,
                success=success,
                status=self.status,
                structured_content=structured,
                content_items=content_items,
                error_message=self.error.message if self.error else None,
                raw=raw,
            )
        )
        return out


class _CollabAgentState(BaseModel):
    """Last known state of a collab agent."""

    model_config = ConfigDict(extra="allow")
    status: str
    message: str | None = None


class _CollabToolCallItem(_CodexItem):
    type: Literal["collab_tool_call"]
    tool: str
    sender_thread_id: str
    receiver_thread_ids: list[str] = Field(default_factory=list)
    prompt: str | None = None
    agents_states: dict[str, _CollabAgentState] = Field(default_factory=dict)
    status: str | None = None

    def translate(self, ctx: _CodexItemContext) -> list[TranscriptEntry]:
        return [
            CollabAgentToolCall(
                id=self.id,
                prompt=self.prompt,
                agents_states={k: v.model_dump(mode="json") for k, v in self.agents_states.items()},
                raw=self._raw(ctx),
            )
        ]


class _WebSearchItem(_CodexItem):
    type: Literal["web_search"]
    query: str
    action: dict[str, Any] | None = None

    def translate(self, ctx: _CodexItemContext) -> list[TranscriptEntry]:
        return [
            WebSearch(
                id=self.id,
                query=self.query,
                action=self.action,
                raw=self._raw(ctx),
            )
        ]


class _TodoListItem(_CodexItem):
    """Running to-do list. Maps to :class:`Plan` with items rendered
    as a markdown checklist; structured list preserved in
    ``raw["_todo_items"]``."""

    type: Literal["todo_list"]
    items: list[dict[str, Any]] = Field(default_factory=list)

    def translate(self, ctx: _CodexItemContext) -> list[TranscriptEntry]:
        lines = []
        for entry in self.items:
            marker = "[x]" if entry.get("completed") else "[ ]"
            text = entry.get("text", "")
            lines.append(f"- {marker} {text}")
        raw = dict(self._raw(ctx))
        raw["_todo_items"] = [dict(item) for item in self.items]
        return [Plan(id=self.id, text="\n".join(lines), raw=raw)]


class _ErrorItem(_CodexItem):
    """Non-fatal error surfaced mid-turn."""

    type: Literal["error"]
    message: str

    def translate(self, ctx: _CodexItemContext) -> list[TranscriptEntry]:
        return [
            TaskNotification(
                task_id=ctx.thread_id or "",
                status="failed",
                summary=self.message,
                output_file="",
                raw=self._raw(ctx),
            )
        ]


class _UserMessageItem(_CodexItem):
    """User-supplied prompt text. Forward-compat for multi-turn captures
    where the CLI echoes prior user prompts back into the stream."""

    type: Literal["user_message"]
    text: str

    def translate(self, ctx: _CodexItemContext) -> list[TranscriptEntry]:
        return [UserPrompt(id=self.id, text=self.text, raw=self._raw(ctx))]


_CodexItemUnion = Annotated[
    _AgentMessageItem
    | _ReasoningItem
    | _CommandExecutionItem
    | _FileChangeItem
    | _McpToolCallItem
    | _CollabToolCallItem
    | _WebSearchItem
    | _TodoListItem
    | _ErrorItem
    | _UserMessageItem,
    Field(discriminator="type"),
]
_CodexItemAdapter: TypeAdapter[_CodexItemUnion] = TypeAdapter(_CodexItemUnion)


# ---------------------------------------------------------------------------
# Envelope-level key sets (no-drop accounting)
# ---------------------------------------------------------------------------

_THREAD_STARTED_CONSUMED: frozenset[str] = frozenset({"type", "thread_id"})
_TURN_COMPLETED_CONSUMED: frozenset[str] = frozenset({"type", "usage"})
_TURN_FAILED_CONSUMED: frozenset[str] = frozenset({"type", "error"})
_ERROR_CONSUMED: frozenset[str] = frozenset({"type", "message"})


def _leftover(d: dict[str, Any], consumed: frozenset[str]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in consumed}


# ---------------------------------------------------------------------------
# CodexDeserializer
# ---------------------------------------------------------------------------


class CodexDeserializer:
    """:class:`TranscriptDeserializer` for the Codex CLI exec JSONL
    stream. Stateless. See :data:`CODEX` for the canonical instance.
    """

    def deserialize(self, raw: list[dict[str, Any]]) -> list[TranscriptEntry]:
        out: list[TranscriptEntry] = []
        thread_id: str | None = None
        for event in raw:
            if not isinstance(event, dict):
                out.append(
                    unknown(
                        "codex",
                        {"_non_dict_event": repr(event)},
                        "non-dict source event",
                    )
                )
                continue
            etype = event.get("type")
            match etype:
                case "thread.started":
                    new_thread_id = event.get("thread_id")
                    thread_id = new_thread_id if isinstance(new_thread_id, str) else None
                    out.append(
                        TaskStarted(
                            task_id=thread_id or "",
                            description="codex thread started",
                            task_type="thread",
                            session_id=thread_id,
                            raw=_leftover(event, _THREAD_STARTED_CONSUMED),
                        )
                    )
                case "turn.started":
                    continue
                case "item.started" | "item.updated":
                    # Superseded by item.completed.
                    continue
                case "item.completed":
                    out.extend(self._translate_item_event(event, thread_id))
                case "turn.completed":
                    usage = event.get("usage")
                    out.append(
                        TaskNotification(
                            task_id=thread_id or "",
                            status="completed",
                            summary="",
                            output_file="",
                            usage=usage if isinstance(usage, dict) else None,
                            session_id=thread_id,
                            raw=_leftover(event, _TURN_COMPLETED_CONSUMED),
                        )
                    )
                case "turn.failed":
                    err = event.get("error")
                    message = ""
                    if isinstance(err, dict):
                        candidate = err.get("message")
                        if isinstance(candidate, str):
                            message = candidate
                    out.append(
                        TaskNotification(
                            task_id=thread_id or "",
                            status="failed",
                            summary=message,
                            output_file="",
                            session_id=thread_id,
                            raw=_leftover(event, _TURN_FAILED_CONSUMED),
                        )
                    )
                case "error":
                    message = ""
                    candidate = event.get("message")
                    if isinstance(candidate, str):
                        message = candidate
                    out.append(
                        TaskNotification(
                            task_id=thread_id or "",
                            status="failed",
                            summary=message,
                            output_file="",
                            session_id=thread_id,
                            raw=_leftover(event, _ERROR_CONSUMED),
                        )
                    )
                case other:
                    out.append(
                        unknown(
                            "codex",
                            dict(event),
                            f"unrecognised ThreadEvent type {other!r}",
                        )
                    )
        return out

    def _translate_item_event(
        self, event: dict[str, Any], thread_id: str | None
    ) -> list[TranscriptEntry]:
        raw_item = event.get("item")
        if not isinstance(raw_item, dict):
            return [
                unknown(
                    "codex",
                    dict(event),
                    "item.completed event missing dict-shaped item",
                )
            ]
        # Envelope leftover (e.g. thread_id we might attach later) becomes
        # the item's overlay. The event itself has only ``type`` and
        # ``item`` keys today, so the overlay is normally empty.
        envelope_extras = {k: v for k, v in event.items() if k not in ("type", "item")}
        overlay: dict[str, Any] = {}
        if thread_id is not None:
            overlay["_thread_id"] = thread_id
        if envelope_extras:
            overlay["_envelope_extras"] = envelope_extras
        ctx = _CodexItemContext(thread_id=thread_id, overlay=overlay)
        try:
            item = _CodexItemAdapter.validate_python(raw_item)
        except ValidationError as exc:
            return [
                unknown(
                    "codex",
                    {"_item": raw_item, "_overlay": overlay},
                    f"unrecognised item type {raw_item.get('type')!r}: {exc.errors()[0]['msg']}",
                )
            ]
        return item.translate(ctx)


CODEX: CodexDeserializer = CodexDeserializer()
"""Canonical :class:`TranscriptDeserializer` for the Codex CLI exec
JSONL wire format."""
