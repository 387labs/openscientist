"""omp (Oh My Pi) :class:`TranscriptDeserializer` backend."""

from __future__ import annotations

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
)
from openscientist.transcript.union import TranscriptEntry
from openscientist.transcript.variants import (
    AssistantText,
    Reasoning,
    ToolCall,
    ToolResult,
    UserPrompt,
)

# Keys mapped to typed fields; everything else is preserved in the raw overlay.
_USER_MSG_CONSUMED: frozenset[str] = frozenset({"role", "content"})
_ASSISTANT_MSG_CONSUMED: frozenset[str] = frozenset({"role", "content", "model"})
_TOOL_RESULT_MSG_CONSUMED: frozenset[str] = frozenset({"role", "toolCallId", "content", "isError"})


def _leftover(d: dict[str, Any], consumed: frozenset[str]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in consumed}


class _OmpBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    def model_extras(self) -> dict[str, Any]:
        return dict(self.__pydantic_extra__ or {})


class _OmpTextBlock(_OmpBlock):
    type: Literal["text"]
    text: str


class _OmpThinkingBlock(_OmpBlock):
    type: Literal["thinking"]
    thinking: str | None = None
    thinking_signature: str | None = Field(default=None, alias="thinkingSignature")


class _OmpToolCallBlock(_OmpBlock):
    type: Literal["toolCall"]
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)  # decoded JSON


_AssistantBlock = Annotated[
    _OmpTextBlock | _OmpThinkingBlock | _OmpToolCallBlock,
    Field(discriminator="type"),
]
_AssistantBlockAdapter: TypeAdapter[_AssistantBlock] = TypeAdapter(_AssistantBlock)


class OmpDeserializer:
    """Translates omp ``agent_end.messages`` into typed entries. See :data:`OMP`."""

    def deserialize(self, raw: list[dict[str, Any]]) -> list[TranscriptEntry]:
        out: list[TranscriptEntry] = []
        for msg in raw:
            if not isinstance(msg, dict):
                out.append(
                    unknown("omp", {"_non_dict_entry": repr(msg)}, "non-dict source message")
                )
                continue
            out.extend(self._translate_message(msg))
        return out

    def _translate_message(self, msg: dict[str, Any]) -> list[TranscriptEntry]:
        role = msg.get("role")
        match role:
            case "user":
                return self._translate_user(msg)
            case "assistant":
                return self._translate_assistant(msg)
            case "toolResult":
                return self._translate_tool_result(msg)
            case other:
                return [unknown("omp", dict(msg), f"unrecognised role {other!r}")]

    def _translate_user(self, msg: dict[str, Any]) -> list[TranscriptEntry]:
        overlay = _leftover(msg, _USER_MSG_CONSUMED)
        content = msg.get("content", [])
        if isinstance(content, str):
            return [UserPrompt(text=content, raw=overlay)]
        if not isinstance(content, list):
            return [unknown("omp", dict(msg), "user message content is neither str nor list")]
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return [UserPrompt(text="\n".join(parts), raw=overlay)]

    def _translate_assistant(self, msg: dict[str, Any]) -> list[TranscriptEntry]:
        model: str | None = msg.get("model")
        overlay = _leftover(msg, _ASSISTANT_MSG_CONSUMED)
        content = msg.get("content", [])
        if not isinstance(content, list):
            return [unknown("omp", dict(msg), "assistant message content is not a list")]
        return [self._translate_assistant_block(block, model, overlay) for block in content]

    def _translate_tool_result(self, msg: dict[str, Any]) -> list[TranscriptEntry]:
        overlay = _leftover(msg, _TOOL_RESULT_MSG_CONSUMED)
        call_id = safe_str(msg.get("toolCallId"), "toolResult.toolCallId")
        content = msg.get("content")
        is_error = bool(msg.get("isError"))
        content_items: list[Any] | None = (
            [dict(item) if isinstance(item, dict) else item for item in content]
            if isinstance(content, list)
            else None
        )
        return [
            ToolResult(
                call_id=call_id,
                output=coerce_tool_result_content(content),
                success=not is_error,
                content_items=content_items,
                raw=overlay,
            )
        ]

    def _translate_assistant_block(
        self,
        raw_block: object,
        model: str | None,
        overlay: dict[str, Any],
    ) -> TranscriptEntry:
        if not isinstance(raw_block, dict):
            return unknown(
                "omp",
                {"_non_dict_block": repr(raw_block), "_overlay": overlay},
                "assistant content block is not a dict",
            )
        try:
            block = _AssistantBlockAdapter.validate_python(raw_block)
        except ValidationError as exc:
            return unknown(
                "omp",
                full_block_overlay(raw_block, overlay),
                f"unrecognised assistant block type {raw_block.get('type')!r}: "
                f"{exc.errors()[0]['msg']}",
            )

        raw = merge_overlay(overlay, block_extras_or_none(block.model_extras()))

        if isinstance(block, _OmpThinkingBlock):
            if not isinstance(block.thinking, str):
                return unknown_block("omp", overlay, block, "thinking block missing thinking text")
            return Reasoning(text=block.thinking, signature=block.thinking_signature, raw=raw)

        if isinstance(block, _OmpToolCallBlock):
            return ToolCall(id=block.id, tool=block.name, arguments=dict(block.arguments), raw=raw)

        if isinstance(block, _OmpTextBlock):
            return AssistantText(text=block.text, model=model, raw=raw)

        return unknown(
            "omp",
            full_block_overlay(raw_block, overlay),
            f"unhandled block type {type(block).__name__!r}",
        )


OMP: OmpDeserializer = OmpDeserializer()
